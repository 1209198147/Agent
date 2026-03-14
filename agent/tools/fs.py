from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.tool import Tool, ToolException


@dataclass(frozen=True)
class _FsConfig:
    base_dir: Path

    def resolve_in_base(self, user_path: str) -> Path:
        """
        Resolve `user_path` under base_dir and block path traversal.
        """
        base = self.base_dir.resolve()
        candidate = (base / user_path).resolve()
        # Python 3.9+: is_relative_to, otherwise fallback
        try:
            if not candidate.is_relative_to(base):
                raise ToolException("path is outside of base_dir")
        except AttributeError:
            if str(candidate).lower().startswith(str(base).lower()) is False:
                raise ToolException("path is outside of base_dir")
        return candidate


class ListDirTool(Tool):
    name = "list_dir"
    description = "List files and folders under a directory (constrained to base_dir)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path relative to base_dir.", "default": "."},
            "max_entries": {"type": "integer", "description": "Max entries to return.", "default": 200},
        },
    }

    def __init__(self, base_dir: str | Path = "."):
        self._cfg = _FsConfig(base_dir=Path(base_dir))

    def call(self, path: str = ".", max_entries: int = 200, **kwargs) -> str:
        target = self._cfg.resolve_in_base(path)
        if not target.exists():
            raise ToolException("list_dir: path does not exist")
        if not target.is_dir():
            raise ToolException("list_dir: path is not a directory")

        entries = []
        for idx, p in enumerate(sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))):
            if idx >= max_entries:
                entries.append("... (truncated)")
                break
            kind = "dir" if p.is_dir() else "file"
            size = p.stat().st_size if p.is_file() else 0
            rel = p.relative_to(self._cfg.base_dir.resolve())
            entries.append(f"{kind}\t{size}\t{rel.as_posix()}")
        return "\n".join(entries)


class ReadTextFileTool(Tool):
    name = "read_text_file"
    description = "Read a UTF-8 text file (constrained to base_dir)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to base_dir."},
            "max_chars": {"type": "integer", "description": "Max characters to return.", "default": 20000},
        },
        "required": ["path"],
    }

    def __init__(self, base_dir: str | Path = "."):
        self._cfg = _FsConfig(base_dir=Path(base_dir))

    def call(self, path: str, max_chars: int = 20000, **kwargs) -> str:
        target = self._cfg.resolve_in_base(path)
        if not target.exists() or not target.is_file():
            raise ToolException("read_text_file: file not found")
        data = target.read_text(encoding="utf-8", errors="replace")
        if len(data) > max_chars:
            return data[:max_chars] + "\n... (truncated)"
        return data


class WriteTextFileTool(Tool):
    name = "write_text_file"
    description = "Write a UTF-8 text file (constrained to base_dir)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to base_dir."},
            "content": {"type": "string", "description": "Text content to write."},
            "mode": {"type": "string", "description": "Write mode: 'overwrite' or 'append'.", "default": "overwrite"},
            "mkdirs": {"type": "boolean", "description": "Create parent directories if needed.", "default": True},
        },
        "required": ["path", "content"],
    }

    def __init__(self, base_dir: str | Path = "."):
        self._cfg = _FsConfig(base_dir=Path(base_dir))

    def call(self, path: str, content: str, mode: str = "overwrite", mkdirs: bool = True, **kwargs) -> str:
        target = self._cfg.resolve_in_base(path)
        if mkdirs:
            target.parent.mkdir(parents=True, exist_ok=True)

        if mode not in {"overwrite", "append"}:
            raise ToolException("write_text_file: mode must be 'overwrite' or 'append'")

        if mode == "append":
            with target.open("a", encoding="utf-8", newline="") as f:
                f.write(content)
        else:
            target.write_text(content, encoding="utf-8", newline="")
        rel = target.relative_to(self._cfg.base_dir.resolve()).as_posix()
        return f"ok: wrote {rel}"

