import json
from typing import AsyncGenerator

from openai import AsyncOpenAI
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from agent.chat_model import ChatModel
from agent.entities import LLMResponse
from agent.tool import ToolSet
from agent.conversation import Conversation
from agent.tool_executor import ToolCallsResult

class OpenAIChatModel(ChatModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url')
        self.timeout = config.get('timeout', 120)
        self.client = AsyncOpenAI(api_key=self.api_key,
                                  base_url=self.base_url,
                                  timeout=self.timeout)

    def _context_to_messages(self, context: Conversation) -> list[dict]:
        messages = []
        for message in context.messages:
            messages.append(message.to_dict())
        return messages

    def _extract_reasoning(self, response: ChatCompletion|ChatCompletionChunk) -> str:
        reasoning = ""
        if len(response.choices) == 0:
            return reasoning
        if isinstance(response, ChatCompletionChunk):
            delta = response.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
        elif isinstance(response, ChatCompletion):
            message = response.choices[0].message
            reasoning = getattr(message, "reasoning_content", None)
        return reasoning if reasoning else ""

    async def _parse_response(self, response: ChatCompletion) -> LLMResponse:
        message = response.choices[0].message
        content = message.content
        model_extra = message.model_extra
        reasoning_content = ""
        if model_extra:
            reasoning = model_extra.get("reasoning_content", None)
            if reasoning:
                reasoning_content = reasoning

        tools_call_args = []
        tools_call_name = []
        tools_call_ids = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tools_call_ids.append(tool_call.id)
                tools_call_name.append(tool_call.function.name)
                tools_call_args.append(json.loads(tool_call.function.arguments))

        return LLMResponse(content=content,
                           reasoning_content=reasoning_content,
                           role=message.role,
                           tools_call_ids=tools_call_ids,
                           tools_call_name=tools_call_name,
                           tools_call_args=tools_call_args,
                           response=response)

    async def _build_chat_payload(self,
                            prompt: str,
                            system_prompt: str,
                            context: Conversation,
                            tool_call_result: ToolCallsResult|list[ToolCallsResult],
                            model: str):
        payload = {}
        if context is None:
            context = Conversation()
        messages = self._context_to_messages(context)

        if system_prompt:
            messages.insert(0, {'role': 'system', 'content': system_prompt})

        if prompt:
            messages.append({'role': 'user', 'content': prompt})

        if tool_call_result:
            if isinstance(tool_call_result, ToolCallsResult):
                messages.extend(tool_call_result.to_open_ai_message())
            else:
                for result in tool_call_result:
                    messages.extend(result.to_open_ai_message())

        payload['messages'] = messages
        payload['model'] = model
        return payload

    async def _query(self, payload, tools: ToolSet|None = None) -> LLMResponse:
        if tools:
            payload['tools'] = tools.to_openai_model()

        response = await self.client.chat.completions.create(
            **payload,
            stream=False
        )
        return await self._parse_response(response)

    async def chat(self,
             prompt: str = None,
             system_prompt: str = None,
             context: Conversation = None,
             tools: ToolSet = None,
             tool_call_result: ToolCallsResult|list[ToolCallsResult] = None,
             model: str = None):
        payload = await self._build_chat_payload(prompt, system_prompt, context, tool_call_result, model)

        return await self._query(payload, tools)

    async def _query_stream(self, payload, tools: ToolSet | None = None) -> AsyncGenerator[LLMResponse, None]:
        if tools:
            payload['tools'] = tools.to_openai_model()

        response_stream = await self.client.chat.completions.create(
            **payload,
            stream=True
        )
        state = ChatCompletionStreamState()
        llm_response = LLMResponse(role="assistant", is_chunk=True)
        async for chunk in response_stream:
            _yield = False
            state.handle_chunk(chunk)
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            reasoning = self._extract_reasoning(chunk)
            if reasoning:
                llm_response.reasoning_content = reasoning
                _yield = True
            if delta.content:
                llm_response.content = delta.content
                _yield = True
            if _yield:
                yield llm_response

        final_completion = state.get_final_completion()
        llm_response = await self._parse_response(final_completion)
        yield llm_response

    async def chat_stream(self,
                          prompt: str = None,
                          system_prompt: str = None,
                          context: Conversation = None,
                          tools: ToolSet = None,
                          tool_call_result: ToolCallsResult|list[ToolCallsResult] = None,
                          model: str = None) -> AsyncGenerator[LLMResponse, None]:
        payload = await self._build_chat_payload(prompt, system_prompt, context, tool_call_result, model)

        async for response in self._query_stream(payload, tools):
            yield response