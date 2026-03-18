from __future__ import annotations
import json
from typing import Any, Literal, ClassVar
from dataclasses import dataclass, field

from agent.utils.file import download_to_base64, file_to_base64

@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str|list[ContentPart]|None = field(default_factory=str)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content if isinstance(self.content, str) else [part.to_dict() for part in self.content]
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
    reasoning_content: str = field(default=None)
    reasoning_signature: str = field(default=None)
    tool_calls: list[ToolCall] = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.reasoning_content:
            d["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            d["tool_calls"] = [tool_call.to_dict() for tool_call in self.tool_calls]
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

class ContentPart:
    __subclasses__: ClassVar[dict[str, type["ContentPart"]]] = {}
    type: Literal["text", "think", "image_url", "audio_url"]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        sub_class_type = getattr(cls, "type", None)
        if not sub_class_type:
            raise ValueError(f"subclass {cls.__name__} must have a type attribute")
        cls.__subclasses__[sub_class_type] = cls

    def to_dict(self) -> dict[str, Any]:
        cls = self.__subclasses__.get(self.type, None)
        if not cls:
            raise ValueError(f"type {self.type} is not supported")
        return cls.to_dict(self)

@dataclass
class TextContentPart(ContentPart):
    type = "text"
    text: str = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "text": self.text
        }

@dataclass
class ThinkContentPart(ContentPart):
    type = "think"
    thinking: str = field(default=None)
    signature: str = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "thinking": self.thinking,
            "signature": self.signature
        }

@dataclass
class ImageContentPart(ContentPart):
    type = "image_url"
    @dataclass
    class ImageUrl:
        url: str
        id: str | None = field(default=None)

        def to_dict(self) -> dict[str, Any]:
            d = {"url": self.url}
            if self.id:
                d["id"] = self.id
            return d
    img_url: ImageUrl
    """base64编码的图片"""

    def __init__(self, url: str, id: str|None=None):
        self.img_url = self.ImageUrl(url=url, id=id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "image_url": self.img_url.to_dict()
        }

@dataclass
class AudioContentPart(ContentPart):
    type = "audio_url"
    class AudioUrl:
        id: str|None = field(default=None)
        url: str
    audio_url: AudioUrl

class DiscardStarStrategy:
    def apply(self, history: list[Message], max_length: int):
        raise NotImplementedError

class DiscardOldestStrategy(DiscardStarStrategy):
    def apply(self, history: list[Message], max_length: int):
        while len(history) > max_length:
            self._discard_oldest_message(history)

    def _discard_oldest_message(self, history: list[Message]):
        """丢掉一个完整的对话消息，如果包括工具调用，对应的工具调用结果也要丢弃"""
        pop_list = []
        idx = 0
        while True:
            message = history[idx]
            _break = True
            if isinstance(message, SystemMessage):
                continue
            elif isinstance(message, AIMessage):
                pop_list.append(idx)
                if message.tool_calls:
                    for _ in message.tool_calls:
                        idx += 1
                        pop_list.append(idx)
            elif isinstance(message, UserMessage):
                pop_list.append(idx)
                _break = False

            if len(pop_list) >=2 and _break:
                break
            idx += 1

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

    @classmethod
    async def assemble_user_message(cls,
                                    prompt: str,
                                    img_urls: list[str]) -> UserMessage:
        content_blocks = []
        if prompt:
            content_blocks.append(TextContentPart(text=prompt))
        elif img_urls:
            content_blocks.append(TextContentPart(text="[图片]"))

        if img_urls:
            for img_url in img_urls:
                if img_url.startswith("http"):
                    _, img_base64, _ = await download_to_base64(url=img_url, encoding="utf-8", ssl_verify=False)
                    content_blocks.append(ImageContentPart(url=f"data:image/jpeg;base64,{img_base64}"))
                elif img_url.startswith("file:///"):
                    img_url = img_url.replace("file:///", "")
                    img_base64 = file_to_base64(img_url, "utf-8")
                    content_blocks.append(ImageContentPart(url=f"data:image/jpeg;base64,{img_base64}"))
                elif img_url.startswith("base64://"):
                    img_url = img_url.replace("base64://", "")
                    content_blocks.append(ImageContentPart(url=f"data:image/jpeg;base64,{img_url}"))
                else:
                    img_base64 = file_to_base64(img_url, "utf-8")
                    content_blocks.append(ImageContentPart(url=f"data:image/jpeg;base64,{img_base64}"))

        if len(content_blocks) == 1 and content_blocks[0].type == "text":
            return UserMessage(content=content_blocks[0].text)
        return UserMessage(content=content_blocks)