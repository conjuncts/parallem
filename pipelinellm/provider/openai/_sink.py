from copy import deepcopy
import json
from typing import List

import polars as pl


def openai_message_sinker(meta: dict, *, remove_content=True):
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


def openai_metadata_sinker(metas: List[tuple[str, str]]):
    """
    Input: List of tuples of (response_id, metadata_json)
    """
    objs = [
        {**as_is, **json.loads(astring)} for as_is, astring in metas if astring.strip()
    ]

    messages_df = None
    # custom handle messages
    messages = []
    for obj in objs:
        my_msg_ids = []
        for msg in obj.get("output", []):
            messages.append(openai_message_sinker(msg, remove_content=True))
            my_msg_ids.append(msg.get("id"))
        obj["output"] = my_msg_ids
    messages_df = pl.DataFrame(messages)

    df = pl.json_normalize(objs)

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
        df, df2 = openai_metadata_sinker([("314", json_file.read())]).values()
    print("Responses DF:")
    print(df)
    print("Messages DF:")
    print(df2)

    # ResponseItem are possibilities for obj["output"][i]
