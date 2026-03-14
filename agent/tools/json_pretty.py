from __future__ import annotations

import json

from agent.tool import Tool, ToolException


class JsonPrettyTool(Tool):
    name = "json_pretty"
    description = "Pretty-print a JSON string."
    parameters = {
        "type": "object",
        "properties": {
            "json_text": {"type": "string", "description": "Raw JSON string."},
            "indent": {"type": "integer", "description": "Indentation spaces.", "default": 2},
            "sort_keys": {"type": "boolean", "description": "Sort object keys.", "default": False},
        },
        "required": ["json_text"],
    }

    def call(self, json_text: str, indent: int = 2, sort_keys: bool = False, **kwargs) -> str:
        try:
            obj = json.loads(json_text)
        except Exception as e:
            raise ToolException(f"json_pretty: invalid JSON ({e})")
        return json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
