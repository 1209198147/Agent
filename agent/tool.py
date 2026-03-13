from typing import Any

class ToolException(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class Tool:
    """工具基类，所有自定义工具应继承此类。"""
    name: str
    description: str
    parameters: dict[str, Any]

    # 用于存储所有 Tool 子类
    _registry: list[type['Tool']] = []

    def __init_subclass__(cls, **kwargs):
        """自动注册子类到注册表。"""
        super().__init_subclass__(**kwargs)
        if cls is not Tool:
            Tool._registry.append(cls)

    def call(self, **kwargs) -> str:
        ...

    def to_dict(self) -> dict[str, str|dict[str, Any]]:
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters,
            }
        }

class ToolSet:
    """工具集类，用于管理多个工具。"""
    tools: list[Tool]

    def __init__(self, tools: list[Tool]):
        self.tools = tools

    def __iter__(self):
        return iter(self.tools)

    def get_tool(self, name: str) -> Tool:
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise ToolException(f"Tool {name} not found")

    def add_tool(self, tool: Tool):
        for idx, _ in enumerate(self.tools):
            if tool.name == _.name:
                self.tools[idx] = tool
                return
        self.tools.append(tool)

    def to_openai_model(self) -> list[dict[str, str|dict[str, Any]]]:
        return [tool.to_dict() for tool in self.tools]