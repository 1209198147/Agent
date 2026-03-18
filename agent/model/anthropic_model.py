import json
from typing import AsyncGenerator

from anthropic import AsyncAnthropic

from agent.chat_model import ChatModel
from agent.conversation import Conversation, AIMessage, UserMessage, ToolCallResult, ImageContentPart, TextContentPart
from agent.entities import LLMResponse
from agent.tool import ToolSet
from agent.tool_executor import ToolCallsResult
from agent.utils.file import download_to_base64, file_to_base64, detect_mime_type


class AnthropicModel(ChatModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url')
        self.timeout = config.get('timeout', 120)
        self.max_tokens = config.get('max_tokens', 4096)
        self.client = AsyncAnthropic(api_key=self.api_key,
                                  base_url=self.base_url,
                                  timeout=self.timeout)


    def _context_to_messages(self, context: Conversation):
        messages = []
        for message in context.get_messages():
            if isinstance(message, UserMessage):
                if isinstance(message.content, str):
                    messages.append({"role": "user", "content": [
                        {
                            "type": "text",
                            "text": message.content
                        }
                    ]})
                elif isinstance(message.content, list):
                    content_blocks = []
                    for part in message.content:
                        if isinstance(part, TextContentPart):
                            content_blocks.append({
                                "type": "text",
                                "text": part.text
                            })
                        elif isinstance(part, ImageContentPart):
                            img_base64 = part.img_url.url
                            img_base64 = img_base64[img_base64.find(",")+1:]
                            content_blocks.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": detect_mime_type(img_base64),
                                    "data": img_base64
                                }
                            })
                    messages.append({"role": "user", "content": content_blocks})
            elif isinstance(message, AIMessage):
                content = []
                if message.content:
                    content.append(
                        {
                            "type": "text",
                            "text": message.content
                        })
                if message.reasoning_content:
                    content.insert(0, {
                        "type": "thinking",
                        "thinking": message.reasoning_content,
                        "signature": message.reasoning_signature
                    })
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "name": tool_call.tool_name,
                            "id": tool_call.tool_call_id,
                            "input": tool_call.arguments
                        })
                messages.append({"role": "assistant", "content": content})
            elif isinstance(message, ToolCallResult):
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": message.content
                        }
                    ]
                })
        return messages

    async def _build_chat_payload(self,
                           prompt: str = None,
                           img_urls: list[str] = None,
                           system_prompt: str = None,
                           context: Conversation = None,
                           tool_call_result: ToolCallsResult|list[ToolCallsResult] = None,
                           model: str = None):
        payload = {}

        if context is None:
            context = Conversation()
        messages = self._context_to_messages(context)

        content_blocks = []
        if prompt:
            content_blocks.append({
                    "type": "text",
                    "text": prompt
                })
        if img_urls:
            for img_url in img_urls:
                if img_url.startswith("http"):
                    _, img_base64, _ = await download_to_base64(url=img_url, encoding="utf-8", ssl_verify=False)
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": detect_mime_type(img_base64),
                            "data": img_base64
                        }
                    })
                elif img_url.startswith("file:///"):
                    img_url = img_url.replace("file:///", "")
                    img_base64 = file_to_base64(img_url, "utf-8")
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": detect_mime_type(img_base64),
                            "data": img_base64
                        }
                    })
                elif img_url.startswith("base64://"):
                    img_url = img_url.replace("base64://", "")
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": detect_mime_type(img_url),
                            "data": img_url
                        }
                    })
                else:
                    img_base64 = file_to_base64(img_url, "utf-8")
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": detect_mime_type(img_base64),
                            "data": img_base64
                        }
                    })


        messages.append({"role": "user", "content": content_blocks})

        if tool_call_result:
            if isinstance(tool_call_result, ToolCallsResult):
                messages.extend(tool_call_result.to_anthropic_message())
            else:
                for result in tool_call_result:
                    messages.extend(result.to_anthropic_message())

        if system_prompt:
            payload["system"] = system_prompt
        payload['messages'] = messages
        payload['model'] = model
        payload['max_tokens'] = self.max_tokens
        return payload


    async def _query(self, payload, tools: ToolSet | None = None) -> LLMResponse:
        if tools:
            payload['tools'] = tools.to_anthropic_model()

        response = await self.client.messages.create(
            **payload,
            stream=False
        )

        llm_response = LLMResponse(role="assistant")

        for content_block in response.content:
            if content_block.type == "text":
                llm_response.content = content_block.text
            if content_block.type == "thinking":
                llm_response.reasoning_content = content_block.thinking
                llm_response.reasoning_signature = content_block.signature
            if content_block.type == "tool_use":
                if not llm_response.tools_call_name:
                    llm_response.tools_call_name = []
                llm_response.tools_call_name.append(content_block.name)
                if not llm_response.tools_call_args:
                    llm_response.tools_call_args = []
                llm_response.tools_call_args.append(content_block.input)
                if not llm_response.tools_call_ids:
                    llm_response.tools_call_ids = []
                llm_response.tools_call_ids.append(content_block.id)

        return llm_response

    async def chat(self,
                   prompt: str = None,
                   img_urls: list[str] = None,
                   system_prompt: str = None,
                   context: Conversation = None,
                   tools: ToolSet = None,
                   tool_call_result: ToolCallsResult|list[ToolCallsResult] = None,
                   model: str = None):
        payload = await self._build_chat_payload(prompt, img_urls, system_prompt, context, tool_call_result, model)
        return await self._query(payload, tools)


    async def _query_stream(self, payload, tools: ToolSet | None = None) -> AsyncGenerator[LLMResponse, None]:
        if tools:
            payload['tools'] = tools.to_anthropic_model()

        stream = await self.client.messages.create(
            **payload,
            stream=True
        )

        final_content = ""
        final_reasoning = ""
        reasoning_signature = ""
        final_tools_call_ids = []
        final_tools_call_names = []
        final_tools_call_arguments = []
        tool_use = dict()
        async for chunk in stream:
            if chunk.type == "message_start":
                pass
            elif chunk.type == "content_block_start":
                if chunk.content_block.type == "tool_use":
                    tool_use[chunk.index] = {
                        "name": chunk.content_block.name,
                        "id": chunk.content_block.id,
                        "input": {}
                    }
            elif chunk.type == "content_block_delta":
                if chunk.delta.type == "text_delta":
                    final_content += chunk.delta.text
                    yield LLMResponse(role="assistant",
                                      content=chunk.delta.text,
                                      is_chunk=True)
                elif chunk.delta.type == "thinking_delta":
                    reasoning_content = chunk.delta.thinking
                    if reasoning_content:
                        final_reasoning += reasoning_content
                        yield LLMResponse(role="assistant",
                                          reasoning_content=reasoning_content,
                                          reasoning_signature=reasoning_signature if reasoning_signature else None,
                                          is_chunk=True)
                elif chunk.delta.type == "signature_delta":
                    reasoning_signature = chunk.delta.signature
                elif chunk.delta.type == "input_json_delta":
                    if chunk.index in tool_use:
                        if "input_json" not in tool_use[chunk.index]:
                            tool_use[chunk.index]["input_json"] = ""
                        tool_use[chunk.index]["input_json"] += chunk.delta.partial_json
            elif chunk.type == "content_block_stop":
                if chunk.index in tool_use:
                    tool_call = tool_use[chunk.index]
                    try:
                        tool_id = tool_call["id"]
                        final_tools_call_ids.append(tool_id)

                        tool_name = tool_call["name"]
                        final_tools_call_names.append(tool_name)


                        if tool_call["input_json"]:
                            tool_call["input"] = json.loads(tool_call["input_json"])
                        tool_args = tool_call["input"]
                        final_tools_call_arguments.append(tool_args)

                        yield LLMResponse(role="tool",
                                          tools_call_ids=[tool_id],
                                          tools_call_name=[tool_name],
                                          tools_call_args=[tool_args],
                                          is_chunk=True)
                    except json.JSONDecodeError:
                        raise ValueError("Invalid JSON in tool call input")
            elif chunk.type == "message_delta":
                pass

        yield LLMResponse(role="assistant",
                          content=final_content or None,
                          reasoning_content=final_reasoning or None,
                          reasoning_signature=reasoning_signature or None,
                          tools_call_ids=final_tools_call_ids or None,
                          tools_call_name=final_tools_call_names or None,
                          tools_call_args=final_tools_call_arguments or None,
                          is_chunk=False)

    async def chat_stream(self,
                          prompt: str = None,
                          img_urls: list[str] = None,
                          system_prompt: str = None,
                          context: Conversation = None,
                          tools: ToolSet = None,
                          tool_call_result: ToolCallsResult|list[ToolCallsResult] = None,
                          model: str = None):
        payload = await self._build_chat_payload(prompt, img_urls, system_prompt, context, tool_call_result, model)
        stream = self._query_stream(payload, tools)
        async for chunk in stream:
            yield chunk
