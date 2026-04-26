from copy import deepcopy
import json
from typing import List

import polars as pl

from parallem.utils.manip import to_snake_case


_schema_overrides = {
    "response_id": pl.Utf8,
    "create_time": pl.Utf8,
    "model_version": pl.Utf8,
    # prompt_feedback
    # automatic_function_calling_history
    # parsed
    "i": pl.Int64,
    "candidates[i].finish_reason": pl.Utf8,
    "candidates[i].content.role": pl.Utf8,
    "candidates[i].rest": pl.Utf8,
    "j": pl.Int64,
    "candidates[i].content.part[j]": pl.Utf8,
    "candidates[i].content.part[j].thought_signature": pl.Utf8,
    "candidates[i].content.part[j].function_call": pl.Utf8,
    "sdk_http_response.headers.content-type": pl.Utf8,
    "sdk_http_response.headers.vary": pl.Utf8,
    "sdk_http_response.headers.content-encoding": pl.Utf8,
    "sdk_http_response.headers.date": pl.Utf8,
    "sdk_http_response.headers.server": pl.Utf8,
    "sdk_http_response.headers.x-xss-protection": pl.Utf8,
    "sdk_http_response.headers.x-frame-options": pl.Utf8,
    "sdk_http_response.headers.x-content-type-options": pl.Utf8,
    "sdk_http_response.headers.server-timing": pl.Utf8,
    "sdk_http_response.headers.alt-svc": pl.Utf8,
    "sdk_http_response.headers.transfer-encoding": pl.Utf8,
    "sdk_http_response.body": pl.Null,
    "usage_metadata.cache_tokens_details": pl.Null,
    "usage_metadata.cached_content_token_count": pl.Int64,  # Null
    "usage_metadata.candidates_token_count": pl.Int64,
    "usage_metadata.candidates_tokens_details": pl.Null,
    "usage_metadata.prompt_token_count": pl.Int64,
    "usage_metadata.prompt_tokens_details": pl.List(
        pl.Struct({"modality": pl.Utf8, "token_count": pl.Int64})
    ),
    "usage_metadata.thoughts_token_count": pl.Int64,
    "usage_metadata.tool_use_prompt_token_count": pl.Int64,  # Null
    "usage_metadata.tool_use_prompt_tokens_details": pl.Null,
    "usage_metadata.total_token_count": pl.Int64,
    "usage_metadata.traffic_type": pl.Null,
}


def compress_google_message(meta: dict, *, remove_content=True):
    # standardize a google message.

    to_string = deepcopy(meta)

    contents = to_string.pop("content", [])
    overall = {
        "i": to_string.pop("index", None),
        "candidates[i].finish_reason": to_string.pop("finish_reason", None),
        "candidates[i].content.role": contents.pop("role", None),
    }
    overall["candidates[i].rest"] = json.dumps(to_string)

    parts = contents.pop("parts", [])
    if not parts:
        yield {
            **overall,
            "j": None,
            "candidates[i].content.part[j]": None,
            "candidates[i].content.part[j].thought_signature": None,
            "candidates[i].content.part[j].function_call": None,
        }
    else:
        for j, part in enumerate(parts):
            if remove_content:
                # if overall["candidates[i].content.role"] == "model":
                part.pop("text", None)

            thought_signature = part.pop("thought_signature", None)
            function_call = part.pop("function_call", None)
            yield {
                **overall,
                "j": j,
                "candidates[i].content.part[j]": json.dumps(part),
                "candidates[i].content.part[j].thought_signature": thought_signature,
                "candidates[i].content.part[j].function_call": json.dumps(function_call)
                if function_call
                else None,
            }


def fix_to_snake_case(obj: dict) -> dict:
    """Recursively converts all keys in a dictionary from camelCase to snake_case."""
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_key = to_snake_case(k)
            new_obj[new_key] = fix_to_snake_case(v)
        return new_obj
    elif isinstance(obj, list):
        return [fix_to_snake_case(item) for item in obj]
    else:
        return obj


def compress_google_metadata(metas: List[str]):
    objs = [
        {**as_is, **json.loads(astring)} for as_is, astring in metas if astring.strip()
    ]

    # custom handle messages
    # assume there's almost always only 1 candidate

    if not objs:
        # Empty
        # TODO: return the right schema
        return {"responses": pl.DataFrame(schema=_schema_overrides)}

    if "modelVersion" in objs[0]:
        # Need to convert camelCase (REST output) to snake_case (python sdk output)
        # Only applies for the batch API, which is smaller (faster, less stuff to process)
        objs = [fix_to_snake_case(obj) for obj in objs]

    request_collector = []
    for obj in objs:
        for msg in obj.pop("candidates", []):
            for part in compress_google_message(msg):
                request_collector.append(
                    {
                        **obj,
                        **part,
                    }
                )

    df = pl.json_normalize(request_collector)

    return {
        "responses": df,
    }
