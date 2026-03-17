from dataclasses import dataclass

from agent.tool_register import tool_manager
from agent.conversation import ToolCall, ToolCallResult, AIMessage
from .tool import ToolException


class ToolCallException(ToolException):
    def __init__(self, tool_name: str, message: str):
        super().__init__(f"Error calling tool '{tool_name}': {message}")

@dataclass
class ToolCallsResult:
    tool_call_info: AIMessage
    tool_call_results: list[ToolCallResult]

    def to_open_ai_message(self) -> list[dict]:
        """转换为 OpenAI 格式的消息"""
        return [
            self.tool_call_info.to_dict(),
            *[result.to_dict() for result in self.tool_call_results]
        ]

    def to_anthropic_message(self) -> list[dict]:
        """转换为 Anthropic 格式的消息"""
        # AI 消息
        ai_content = []
        if self.tool_call_info.content:
            ai_content.append({
                "type": "text",
                "text": self.tool_call_info.content
            })
        if self.tool_call_info.tool_calls:
            for tool_call in self.tool_call_info.tool_calls:
                ai_content.append({
                    "type": "tool_use",
                    "name": tool_call.tool_name,
                    "id": tool_call.tool_call_id,
                    "input": tool_call.arguments
                })
        
        messages = [{"role": "assistant", "content": ai_content}]
        
        # 工具调用结果
        for result in self.tool_call_results:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result.tool_call_id,
                        "content": result.content
                    }
                ]
            })
        return messages

class ToolExecutor:
    """工具执行器"""
    def call(self, tool_calls: list[ToolCall]) -> ToolCallsResult:
        tool_call_results = [self._call(tool_call) for tool_call in tool_calls]
        tool_call_info = AIMessage(tool_calls=tool_calls)
        return ToolCallsResult(tool_call_info=tool_call_info, tool_call_results=tool_call_results)

    def _call(self, tool_call_info: ToolCall) -> ToolCallResult:
        tool = tool_manager.get_tool(tool_call_info.tool_name)
        args = tool_call_info.arguments
        try:
            result = tool.call(**args)
            return ToolCallResult(tool_call_id=tool_call_info.tool_call_id, content=result)
        except Exception as e:
            return ToolCallResult(tool_call_id=tool_call_info.tool_call_id, content=str(e))