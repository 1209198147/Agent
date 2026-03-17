from typing import Any
from openai.types.chat import ChatCompletion

from agent.conversation import ToolCall


class LLMResponse:
    """
    LLMResponse is a class that represents the response from the LLM.
    """
    role: str
    content: str
    reasoning_content: str
    reasoning_signature: str
    tools_call_args: list[dict[str, Any]]
    tools_call_name: list[str]
    tools_call_ids: list[str]
    is_chunk: bool

    raw_response: ChatCompletion

    def __init__(self, response: ChatCompletion|None = None,
                 role: str|None = None,
                 content: str|None = None,
                 reasoning_content: str|None = None,
                 reasoning_signature: str|None = None,
                 tools_call_args: list[dict[str, Any]]|None = None,
                 tools_call_name: list[str]|None = None,
                 tools_call_ids: list[str]|None = None,
                 is_chunk: bool = False):
        self.raw_response = response
        self.role = role
        self.reasoning_content = reasoning_content
        self.reasoning_signature = reasoning_signature
        self.content = content
        self.tools_call_args = tools_call_args
        self.tools_call_name = tools_call_name
        self.tools_call_ids = tools_call_ids
        self.is_chunk = is_chunk

    def get_tools_call(self) -> list[ToolCall]:
        tools_call = []
        for idx in range(len(self.tools_call_ids)):
            tools_call.append(ToolCall(
                tool_call_id=self.tools_call_ids[idx],
                tool_name=self.tools_call_name[idx],
                arguments=self.tools_call_args[idx]
            ))
        return tools_call