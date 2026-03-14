from __future__ import annotations

from pathlib import Path

from agent.tool import Tool, ToolSet

from .calc import CalcTool
from .done import DoneTool
from .echo import EchoTool
from .fs import ListDirTool, ReadTextFileTool, WriteTextFileTool
from .json_pretty import JsonPrettyTool
from .time import TimeTool
from .web import WebLinksTool, WebOpenTool, WebSearchTool


def default_tools(base_dir: str | Path = ".") -> list[Tool]:
    """
    Convenience factory for a "reasonable" default tool list.

    `base_dir` constrains filesystem tools to a subtree, preventing accidental
    access outside that directory.
    """
    root = Path(base_dir).resolve()
    return [
        DoneTool(),
        EchoTool(),
        TimeTool(),
        CalcTool(),
        JsonPrettyTool(),
        WebSearchTool(),
        WebOpenTool(),
        WebLinksTool(),
        ListDirTool(base_dir=root),
        ReadTextFileTool(base_dir=root),
        WriteTextFileTool(base_dir=root),
    ]


def default_toolset(base_dir: str | Path = ".") -> ToolSet:
    return ToolSet(default_tools(base_dir=base_dir))

