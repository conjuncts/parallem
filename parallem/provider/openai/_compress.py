from copy import deepcopy
import json
from typing import List

import polars as pl

from parallem.core.compress._json_normalize import pl_json_normalize


def compress_openai_message(meta: dict, *, remove_content=True):
    # standardize an openai message.
    # See from openai.types.responses.response_item import ResponseItem

    to_string = deepcopy(meta)

    overall = {
        "id": to_string.pop("id", None),
        "type": to_string.pop("type", None),
        "status": to_string.pop("status", None),
        "role": to_string.pop("role", None),
    }

    if remove_content:
        if overall["type"] == "message":
            for item in to_string.get("content") or []:
                item.pop("text", None)

    return {
        **overall,
        "rest": json.dumps(to_string),
    }


def compress_openai_metadata(metas: List[tuple[str, str]]):
    """
    Input: List of tuples of (response_id, metadata_json)
    """
    objs = [{**as_is, **json.loads(astring)} for as_is, astring in metas if astring.strip()]

    messages_df = None
    # custom handle messages
    messages = []
    for obj in objs:
        my_msg_ids = []
        for msg in obj.get("output", []):
            messages.append(compress_openai_message(msg, remove_content=True))
            my_msg_ids.append(msg.get("id"))
        obj["output"] = my_msg_ids
    messages_df = pl.DataFrame(
        messages,
        schema={
            "id": pl.Utf8,
            "type": pl.Utf8,
            "status": pl.Utf8,
            "role": pl.Utf8,
            "rest": pl.Utf8,
        },
    )

    # df = pl.json_normalize(objs)
    # https://developers.openai.com/api/reference/resources/responses
    # from openai.types.responses.response import Response
    df = pl_json_normalize(
        objs,
        schema_overrides={
            "response_id": pl.Utf8,
            "object": pl.Utf8,
            "created_at": pl.Int64,
            # error
            "error.code": pl.Utf8,
            "error.message": pl.Utf8,
            "status": pl.Utf8,
            "background": pl.Boolean,
            "completed_at": pl.Int64,
            "frequency_penalty": pl.Float64,
            # incomplete_details
            "incomplete_details.reason": pl.Utf8,
            # "instructions": pl.Null,
            "max_output_tokens": pl.Int64,
            "max_tool_calls": pl.Int64,
            "model": pl.Utf8,
            # "moderation": pl.Null,
            "output": pl.List(pl.Utf8),
            "parallel_tool_calls": pl.Boolean,
            "presence_penalty": pl.Float64,
            "previous_response_id": pl.Utf8,
            "prompt_cache_key": pl.Utf8,
            "prompt_cache_retention": pl.Utf8,
            "safety_identifier": pl.Utf8,
            "service_tier": pl.Utf8,
            "store": pl.Boolean,
            "temperature": pl.Float64,
            "tool_choice": pl.Utf8,
            # "tools": pl.List(pl.Null),
            "top_logprobs": pl.Int64,
            "top_p": pl.Float64,
            "truncation": pl.Utf8,
            "user": pl.Utf8,
            "billing.payer": pl.Utf8,
            "reasoning.effort": pl.Utf8,
            "reasoning.summary": pl.Utf8,
            "text.format.type": pl.Utf8,
            "text.verbosity": pl.Utf8,
            "usage.input_tokens": pl.Int64,
            "usage.input_tokens_details.cached_tokens": pl.Int64,
            "usage.output_tokens": pl.Int64,
            "usage.output_tokens_details.reasoning_tokens": pl.Int64,
            "usage.total_tokens": pl.Int64,
        },
    )

    return {
        "responses": df,
        "messages": messages_df,
    }


if __name__ == "__main__":
    # Debug out the openai schema

    from openai.types.responses.response_item import ResponseItem

    r: ResponseItem = None

    with open(
        "experiments/schema/openai_metadata_example.json", "r", encoding="utf-8"
    ) as json_file:
        df, df2 = compress_openai_metadata([("314", json_file.read())]).values()
    print("Responses DF:")
    print(df)
    print("Messages DF:")
    print(df2)

    # ResponseItem are possibilities for obj["output"][i]
