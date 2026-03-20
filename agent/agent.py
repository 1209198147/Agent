import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator

from agent.chat_model import ChatModel
from agent.context import AgentContext
from agent.conversation import Conversation, ToolCall, AIMessage, UserMessage, ToolCallResult
from agent.entities import LLMResponse
from agent.tool import Tool, ToolSet
from agent.tool_executor import ToolExecutor

DONE = 'done'

class AgentStatus(Enum):
    IDLE = 0
    RUNNING = 1
    STOP = 2
    DONE = 3
    ERROR = 4

@dataclass
class AgentResponse:
    """Agent响应基类"""
    content: str|None = field(default=None)
    reasoning_content: str|None = field(default=None)
    tool_calls: list[ToolCall]|None = field(default=None)
    tool_calls_result: list[ToolCallResult]|None= field(default=None)
    is_chunk: bool = field(default=False)

class Agent:
    """Agent基类，定义智能体的基本行为"""
    name: str
    model: ChatModel
    model_name: str
    status: AgentStatus
    chat_history :Conversation
    
    def __init__(self, name: str, model: ChatModel, model_name:str, chat_history: Conversation):
        self.name = name
        self.model = model
        self.model_name = model_name
        self.status = AgentStatus.IDLE
        self.chat_history = chat_history
        self.status = AgentStatus.IDLE

    def run(self, *args, **kwargs):
        """运行Agent"""
        raise NotImplementedError("run method must be implemented")

    def start(self):
        """启动Agent"""
        self.status = AgentStatus.RUNNING
    
    def stop(self):
        """停止Agent"""
        self.status = AgentStatus.STOP

    def done(self):
        """完成任务"""
        self.status = AgentStatus.DONE

    def is_stop(self) -> bool:
        """检查Agent是否停止"""
        return self.status == AgentStatus.STOP
    
    def reset(self):
        """重置Agent状态"""
        self.status = AgentStatus.IDLE
    
    def is_running(self) -> bool:
        """检查Agent是否正在运行"""
        return self.status == AgentStatus.RUNNING

    def is_done(self):
        """检查Agent是否完成任务"""
        return self.status == AgentStatus.DONE

class ReActAgent(Agent):
    """ReAct模式的Agent，实现Reasoning + Acting循环"""
    
    def __init__(self, name: str,
                 model: ChatModel,
                 model_name: str,
                 chat_history: Conversation,
                 **kwargs):
        super().__init__(name, model, model_name, chat_history)

    async def run(self, context: AgentContext, max_steps: int=20) -> None|AsyncGenerator[AgentResponse, None]:
        """执行ReAct循环"""
        self.start()
        cur_step = 0
        while not self.is_stop() and cur_step < max_steps:
            reason = self._reason(context)
            if isinstance(reason, AsyncGenerator):
                async for response in reason:
                    yield response
            else:
                yield reason
            act = self._act(context)
            if isinstance(act, AsyncGenerator):
                async for response in act:
                    yield response
            else:
                yield act
            cur_step += 1
            if self.is_stop() or self.is_done():
                break

    async def _reason(self, context: AgentContext) -> None|AsyncGenerator[AgentResponse, None]:
        """Reasoning阶段"""
        ...

    async def _act(self, context: AgentContext) -> None|AsyncGenerator[AgentResponse, None]:
        """Acting阶段"""
        ...

class ToolCallAgent(ReActAgent):
    tools: list[Tool]|ToolSet
    tool_executor: ToolExecutor
    def __init__(self,
                 name: str,
                 model: ChatModel,
                 model_name: str,
                 chat_history: Conversation,
                 tools: list[Tool]|ToolSet,
                 tool_executor: ToolExecutor):
        super().__init__(name, model, model_name, chat_history)
        self.tools = tools
        self.tool_executor = tool_executor

    async def _reason(self, context: AgentContext) -> None|AsyncGenerator[AgentResponse, None]:
        """Reasoning阶段：调用LLM获取响应"""
        # 添加用户消息（只在第一轮添加）
        if context.user_prompt and not context.results:
            self.chat_history.add_message(Conversation.assemble_user_message(context.user_prompt, context.img_urls))
            context.user_prompt = ''  # 清空，避免重复添加
        llm_response: LLMResponse = await self.model.chat(
            system_prompt=context.system_prompt,
            context=self.chat_history,
            tools=self.tools if isinstance(self.tools, ToolSet) else ToolSet(self.tools),
            model=self.model_name
        )
        context.llm_response = llm_response

        self.chat_history.add_message(AIMessage(content=llm_response.content,
                                                reasoning_content=llm_response.reasoning_content,
                                                tool_calls=llm_response.get_tools_call()))
        context.tool_calls = llm_response.get_tools_call()
        context.results.append(llm_response)
        yield AgentResponse(content=llm_response.content,
                            reasoning_content=llm_response.reasoning_content,
                            tool_calls=context.tool_calls,
                            is_chunk=llm_response.is_chunk)

    async def _act(self, context: AgentContext) -> None|AsyncGenerator[AgentResponse, None]:
        """Acting阶段：处理工具调用"""
        tool_calls = context.tool_calls

        if tool_calls:
            # 有工具调用
            for tool_call in tool_calls:
                # 检查AI是否调用了done工具，如果调用了则代表任务已经完成
                if tool_call.tool_name == DONE:
                    self.done()
            # 执行工具调用
            tool_calls_result = self.tool_executor.call_batch(tool_calls)
            if context.llm_response.reasoning_content:
                tool_calls_result.tool_call_info.reasoning_content = context.llm_response.reasoning_content
            context.tool_calls_result = tool_calls_result
            for tool_call_result in tool_calls_result.tool_call_results:
                self.chat_history.add_message(tool_call_result)
            yield AgentResponse(tool_calls_result=tool_calls_result.tool_call_results)
        else:
            # 如果没有工具调用，认为ai已经完成了任务
            self.done()


class StreamToolCallAgent(ReActAgent):
    """流式输出的ToolCallAgent，支持实时展示AI响应"""
    tools: list[Tool] | ToolSet
    tool_executor: ToolExecutor

    def __init__(self,
                 name: str,
                 model: ChatModel,
                 model_name: str,
                 chat_history: Conversation,
                 tools: list[Tool] | ToolSet,
                 tool_executor: ToolExecutor):
        super().__init__(name, model, model_name, chat_history)
        self.tools = tools
        self.tool_executor = tool_executor

    async def _reason_stream(self, context: AgentContext) -> AsyncGenerator[AgentResponse, None]:
        """
        流式Reasoning阶段：调用LLM获取流式响应
        使用AsyncGenerator逐步产出响应，实现实时输出
        最后一个chunk包含完整的响应信息
        """
        # 添加用户消息（只在第一轮添加）
        if context.user_prompt and not context.results:
            self.chat_history.add_message(UserMessage(content=context.user_prompt))
            context.user_prompt = ''  # 清空，避免重复添加

        stream = self.model.chat_stream(
            system_prompt=context.system_prompt,
            context=self.chat_history,
            tools=self.tools if isinstance(self.tools, ToolSet) else ToolSet(self.tools),
            model=self.model_name
        )

        llm_response = None
        async for chunk in stream:
            response = AgentResponse(content=chunk.content,
                                     reasoning_content=chunk.reasoning_content, is_chunk=True)
            if not chunk.is_chunk:
                llm_response = chunk
                response.tool_calls = chunk.get_tools_call()
                response.is_chunk = False
            yield response

        self.chat_history.add_message(AIMessage(
            content=llm_response.content,
            reasoning_content=llm_response.reasoning_content,
            tool_calls=llm_response.get_tools_call()
        ))
        context.llm_response = llm_response
        context.tool_calls = llm_response.get_tools_call()
        context.results.append(llm_response)

    async def _act_async(self, context: AgentContext) -> AsyncGenerator[AgentResponse, None]:
        """异步Acting阶段：处理工具调用"""
        tool_calls = context.tool_calls
        if tool_calls:
            # 有工具调用
            for tool_call in tool_calls:
                # 检查AI是否调用了done工具，如果调用了则代表任务已经完成
                if tool_call.tool_name == DONE:
                    self.done()
                    continue
                tool_call_result = self.tool_executor.call(tool_call)
                self.chat_history.add_message(tool_call_result)
                yield AgentResponse(tool_calls_result=[tool_call_result])
        else:
            # 如果没有工具调用，认为ai已经完成了任务
            self.done()

    async def run_stream(self, context: AgentContext, max_steps: int = 20) -> AsyncGenerator[AgentResponse, None]:
        """
        流式执行ReAct循环
        """
        self.start()
        cur_step = 0
        while not self.is_stop() and cur_step < max_steps:
            # 流式输出reasoning阶段
            async for chunk in self._reason_stream(context):
                yield chunk

            # 执行acting阶段
            async for chunk in self._act_async(context):
                yield chunk

            cur_step += 1
            if self.is_stop() or self.is_done():
                break