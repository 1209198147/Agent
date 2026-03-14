"""
Builtin tool implementations for the agent framework.

Importing this package makes the Tool subclasses available for manual wiring.
The framework does not auto-register tools unless you explicitly construct a
ToolSet (recommended) or call tool_manager.initialization() after importing.
"""

from .done import DoneTool
from .echo import EchoTool
from .time import TimeTool
from .calc import CalcTool
from .json_pretty import JsonPrettyTool
from .fs import ListDirTool, ReadTextFileTool, WriteTextFileTool
from .web import WebLinksTool, WebOpenTool, WebSearchTool
from .defaults import default_tools, default_toolset

__all__ = [
    "DoneTool",
    "EchoTool",
    "TimeTool",
    "CalcTool",
    "JsonPrettyTool",
    "ListDirTool",
    "ReadTextFileTool",
    "WriteTextFileTool",
    "WebSearchTool",
    "WebOpenTool",
    "WebLinksTool",
    "default_tools",
    "default_toolset",
]

