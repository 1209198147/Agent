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
        return [
            self.tool_call_info.to_dict(),
            *[result.to_dict() for result in self.tool_call_results]
        ]

class ToolExecutor:

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