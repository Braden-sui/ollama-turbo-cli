# Third-party plugin: hello

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "hello",
        "description": "Say hello, optionally addressing a person by name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Optional name to greet"}
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}


def hello(name: str | None = None) -> str:
    if name and str(name).strip():
        return f"Hello, {name}! 👋"
    return "Hello, world! 👋"
