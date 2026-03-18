from typing import AsyncGenerator

from agent.conversation import Conversation
from agent.entities import LLMResponse
from agent.tool import ToolSet
from agent.tool_executor import ToolCallsResult


class ChatModel:
    config: dict

    def __init__(self, config: dict):
        self.config = config

    async def chat(self,
                   prompt: str = None,
                   img_urls: list[str] = None,
                   system_prompt: str = None,
                   context: Conversation = None,
                   tools: ToolSet = None,
                   tool_call_result: ToolCallsResult|list[ToolCallsResult] = None,
                   model: str = None):
        ...

    async def chat_stream(self,
                          prompt: str = None,
                          img_urls: list[str] = None,
                          system_prompt: str = None,
                          context: Conversation = None,
                          tools: ToolSet = None,
                          tool_call_result: ToolCallsResult|list[ToolCallsResult] = None,
                          model: str = None) -> AsyncGenerator[LLMResponse, None]:
        ...