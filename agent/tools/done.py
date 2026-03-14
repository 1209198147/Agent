from agent.tool import Tool


class DoneTool(Tool):
    """
    Signals the agent loop that the task is complete.

    The framework checks `tool_name == "done"` to stop the loop.
    """

    name = "done"
    description = "Mark the current task as completed."
    parameters = {"type": "object", "properties": {}}

    def call(self, **kwargs) -> str:
        return "done"

