from typing import Any

from agent.entities import LLMResponse
from agent.conversation import ToolCall


class AgentContext:
    extra: dict[str, Any]
    def __init__(self, extra=None):
        if extra is None:
            extra = {}
        self.extra = extra
        self.llm_response: LLMResponse|None = None
        self.tool_calls: list[ToolCall]|None = None
        self.results: list[LLMResponse] = []
        self.system_prompt = ''
        self.user_prompt = ''

    def clear(self):
        self.extra = {}
        self.llm_response = None
        self.results = []
        self.system_prompt = ''
        self.user_prompt = ''