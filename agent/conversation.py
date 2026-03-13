import json
from typing import Any, Literal
from dataclasses import dataclass, field

@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = field(default_factory=str)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content
        }

@dataclass
class ToolCall:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.tool_call_id,
            "type": "function",
            "function": {
                "name": self.tool_name,
                "arguments": json.dumps(self.arguments)
            }
        }

@dataclass
class AIMessage(Message):
    role: Literal["assistant"] = "assistant"
    tool_calls: list[ToolCall] = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.tool_calls:
            d["tool_calls"] = [tool_call.to_dict() for tool_call in self.tool_calls]
        return d

@dataclass
class AIReasoningMessage(AIMessage):
    reasoning_content: str = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.reasoning_content:
            d["reasoning_content"] = self.reasoning_content
        return d

@dataclass
class SystemMessage(Message):
    role: Literal["system"] = "system"

@dataclass
class UserMessage(Message):
    role: Literal["user"] = "user"

@dataclass
class ToolCallResult(Message):
    role: Literal["tool"] = "tool"
    tool_call_id: str = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d

class DiscardStarStrategy:
    def apply(self, history: list[Message], max_length: int):
        raise NotImplementedError

class DiscardOldestStrategy(DiscardStarStrategy):
    def apply(self, history: list[Message], max_length: int):
        while len(history) > max_length:
            self._discard_oldest_message(history)

    def _discard_oldest_message(self, history: list[Message]):
        pop_list = []
        pop_count = 0
        for idx, message in enumerate(history):
            if message.role == "system":
                continue
            pop_list.append(idx)
            pop_count += 1
            if pop_count == 2:
                break
        for idx in reversed(pop_list):
            history.pop(idx)

class Conversation:
    def __init__(self, max_length: int=20):
        self.messages: list[Message] = []
        self.discard_strategy: DiscardStarStrategy = DiscardOldestStrategy()
        self.max_length = max_length

    def add_message(self, message: Message):
        self.messages.append(message)
        self.discard_strategy.apply(self.messages, self.max_length)

    def get_messages(self) -> list[Message]:
        return self.messages