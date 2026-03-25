import json
from typing import List

from parallem.types import FunctionCall


def dump_function_calls(function_calls: List[FunctionCall]) -> str:
    """
    Serialize a list of FunctionCall objects into a JSON-ified list of tuples.
    """
    if function_calls is None:
        return None
    return json.dumps([[call.name, call.args, call.call_id] for call in function_calls])


def load_function_calls(data: str) -> List[FunctionCall]:
    """
    Deserialize a list of tuples into a list of FunctionCall objects.
    """
    tuples = json.loads(data)
    return [
        FunctionCall(name=name, arguments=arguments, call_id=call_id)
        for name, arguments, call_id in tuples
    ]
