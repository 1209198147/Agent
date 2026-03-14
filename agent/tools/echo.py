from agent.tool import Tool


class EchoTool(Tool):
    name = "echo"
    description = "Echo back the given text."
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to echo back."}
        },
        "required": ["text"],
    }

    def call(self, text: str, **kwargs) -> str:
        return text
