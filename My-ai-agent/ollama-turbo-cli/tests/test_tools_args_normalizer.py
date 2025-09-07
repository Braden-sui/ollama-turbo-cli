import json
import random
import string
import pytest

try:
    import jsonschema  # type: ignore
    HAS_JSONSCHEMA = True
except Exception:
    HAS_JSONSCHEMA = False

from src.tools_runtime.args import normalize_args


def test_normalize_args_parses_json_and_dict():
    schema = None
    # dict passthrough
    d = {"x": 1, "name": "a"}
    assert normalize_args(schema, d) == d
    # json string parse
    s = json.dumps(d)
    assert normalize_args(schema, s) == d
    # empty string -> {}
    assert normalize_args(schema, " ") == {}
    # None -> {}
    assert normalize_args(schema, None) == {}


def test_normalize_args_rejects_non_object():
    with pytest.raises(ValueError):
        normalize_args(None, 123)
    with pytest.raises(ValueError):
        normalize_args(None, "not json")


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_normalize_args_validates_schema():
    schema = {
        "type": "object",
        "required": ["x", "name"],
        "properties": {
            "x": {"type": "integer"},
            "name": {"type": "string"},
        },
        "additionalProperties": False,
    }
    ok = {"x": 1, "name": "ok"}
    assert normalize_args(schema, ok) == ok
    # bad type
    with pytest.raises(ValueError):
        normalize_args(schema, {"x": "1", "name": "ok"})
    # missing field
    with pytest.raises(ValueError):
        normalize_args(schema, {"x": 2})
    # extra field
    with pytest.raises(ValueError):
        normalize_args(schema, {"x": 2, "name": "ok", "y": 3})


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_normalize_args_fuzz_validation():
    schema = {
        "type": "object",
        "required": ["x", "name"],
        "properties": {
            "x": {"type": "integer"},
            "name": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
    }
    def rand_str():
        return ''.join(random.choice(string.ascii_letters) for _ in range(random.randint(1, 8)))
    for _ in range(50):
        # Randomly omit or flip types
        args = {}
        if random.random() < 0.7:
            args["x"] = random.randint(-5, 5) if random.random() < 0.5 else str(random.randint(-5, 5))
        if random.random() < 0.7:
            args["name"] = rand_str() if random.random() < 0.5 else random.randint(0, 9)
        # Maybe add extra prop
        if random.random() < 0.3:
            args["extra"] = 1
        if set(args.keys()) == {"x", "name"} and isinstance(args["x"], int) and isinstance(args["name"], str) and len(args["name"]) > 0:
            # Valid case should pass
            assert normalize_args(schema, args) == args
        else:
            with pytest.raises(ValueError):
                normalize_args(schema, args)
