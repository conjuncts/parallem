import inspect
from typing import Callable, List, Optional, Union, get_origin, get_args


def python_type_to_json_schema(tp):
    origin = get_origin(tp)
    args = get_args(tp)

    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is str:
        return {"type": "string"}
    if tp is bool:
        return {"type": "boolean"}
    if origin is list:
        return {"type": "array", "items": python_type_to_json_schema(args[0])}
    if origin is Optional:
        return python_type_to_json_schema(args[0])

    return {"type": "string"}  # fallback


def to_tool_schema(funcs: Union[Callable, List[Callable]]) -> List[dict]:
    """Turns a list of functions into a list of OpenAPI-style JSON tool schemas, for use in tool calls."""
    tool_schemas = []

    if not isinstance(funcs, (list, tuple)):
        funcs = [funcs]
    for func in funcs:
        sig = inspect.signature(func)
        params = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        for name, param in sig.parameters.items():
            params["properties"][name] = python_type_to_json_schema(param.annotation)
            if param.default is inspect.Parameter.empty:
                params["required"].append(name)

        tool_schemas.append(
            {
                "type": "function",
                "name": func.__name__,
                "description": func.__doc__ or "",
                "parameters": params,
            }
        )
    return tool_schemas
