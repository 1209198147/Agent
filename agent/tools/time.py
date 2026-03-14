from __future__ import annotations

from datetime import datetime, timezone

from agent.tool import Tool


class TimeTool(Tool):
    name = "time_now"
    description = "Get current time in ISO-8601 (local and UTC)."
    parameters = {"type": "object", "properties": {}}

    def call(self, **kwargs) -> str:
        local = datetime.now().astimezone()
        utc = datetime.now(timezone.utc)
        return f"local={local.isoformat()} utc={utc.isoformat()}"
