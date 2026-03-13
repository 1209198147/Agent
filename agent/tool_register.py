from agent.tool import Tool, ToolException, ToolSet


class ToolNotFoundException(ToolException):
    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' not found")

class ToolDict(dict):
    """用于管理工具的字典类，支持属性访问方式。"""

    def __getattr__(self, item: str) -> Tool:
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'ToolDict' object has no attribute '{item}'")

    def __setattr__(self, key: str, value: Tool):
        if not isinstance(value, Tool):
            raise TypeError("value must be a Tool")
        self[key] = value

    def __delattr__(self, key: str):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'ToolDict' object has no attribute '{key}'")

    def get_tools(self) -> ToolSet:
        """获取所有工具列表。"""
        return ToolSet(list(self.values()))

class ToolManager:
    def __init__(self):
        self.tools: ToolDict = ToolDict()

    def initialization(self, tools: list[Tool] | ToolSet | None = None):
        """初始化工具管理器，注册所有 Tool 子类实例。

        Args:
            tools: 可选的工具实例列表。如果为 None，则注册所有已注册子类的默认实例。
        """
        if tools is not None:
            for tool in tools:
                self.register(tool)
        else:
            # 自动实例化并注册所有 Tool 子类
            for tool_cls in Tool._registry:
                try:
                    # 尝试创建默认实例（需要子类提供默认构造参数）
                    tool_instance = tool_cls()
                    self.register(tool_instance)
                except TypeError:
                    # 如果子类没有无参构造函数，跳过
                    pass


    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get_tool(self, name) -> Tool:
        if name not in self.tools:
            raise ToolNotFoundException(name)
        return self.tools[name]

    def get_tools(self, tool_names: list[str]|tuple[str]|set[str]|None = None) -> ToolSet:
        if tool_names is None:
            return self.tools.get_tools()
        for tool_name in tool_names:
            if tool_name not in self.tools:
                raise ToolNotFoundException(tool_name)
        return ToolSet([self.tools[tool_name] for tool_name in tool_names])

    def clear(self):
        self.tools.clear()

tool_manager = ToolManager()