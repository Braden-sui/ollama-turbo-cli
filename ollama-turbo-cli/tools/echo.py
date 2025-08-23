# Third-party plugin: echo

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Echo back the provided text, with optional transformations.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to echo back"},
                "uppercase": {"type": "boolean", "description": "Return text in uppercase"},
                "reverse": {"type": "boolean", "description": "Return text reversed"}
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
}


def echo(text: str, uppercase: bool = False, reverse: bool = False) -> str:
    out = text
    if uppercase:
        out = out.upper()
    if reverse:
        out = out[::-1]
    return out
