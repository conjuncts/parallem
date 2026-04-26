import json
import os
from pathlib import Path

import polars as pl

from parallem.core.compress._json_normalize import pl_json_normalize

_schema_overrides = {
    "custom_id": pl.Utf8,
    "method": pl.Utf8,
    "url": pl.Utf8,
    "body.model": pl.Utf8,
    # "body.instructions": pl.Utf8,
    # "body.input": pl.List(), --> needs to be stored as bytes, since it can be arbitrarily nested.
    # "body.reasoning.effort": pl.Utf8,
    # "body.max_output_tokens": pl.Int64,
}

_MISSING = object()


def _pop_json_or_none(container, key):
    """
    Pop and JSON-serialize a key only if it exists.

    Convention:
    - None means the key was absent.
    - "null" means the key existed and was JSON null.

    :param container: Dict-like object.
    :param key: Key to pop.
    :return: JSON string or None.
    """
    if key not in container:
        return None
    return json.dumps(container.pop(key))


def pack_openai_item(item):
    """
    Warning: mutates the input item.

    """
    custom_id = item.pop("custom_id")
    method = item.pop("method")
    url = item.pop("url")
    body = item.pop("body")

    body_model = body.pop("model")
    body_instructions = body.pop("instructions")

    body_input = json.dumps(body.pop("input"))

    body_tools = json.dumps(body.pop("tools"))

    body_text_format = None
    if "text" in body and isinstance(body["text"], dict):
        body_text_format = _pop_json_or_none(body["text"], "format")
        if not body["text"]:
            # remove if empty after popping format
            body.pop("text")

    body_rest = json.dumps(body) if body else None
    item_rest = json.dumps(item) if item else None
    return {
        "custom_id": custom_id,
        "method": method,
        "url": url,
        "body.model": body_model,
        "body.instructions": body_instructions,
        "body.input": body_input,
        "body.tools": body_tools,
        "body.text.format": body_text_format,
        "body.rest": body_rest,
        "item.rest": item_rest,
    }


def _json_loads_if_not_none(value):
    """
    Deserialize a JSON string if present.

    :param value: JSON string or None.
    :return: Parsed Python value or None.
    """
    if value is None:
        return None
    return json.loads(value)


def _pop_unpacked_json_or_missing(container, key):
    """
    Pop and decode a packed JSON field while preserving missing state.

    Convention:
    - Missing key or None value => missing field.
    - "null" => present field with None.

    :param container: Packed dict-like object.
    :param key: Key to pop.
    :return: Parsed value or _MISSING sentinel.
    """
    raw = container.pop(key, _MISSING)
    if raw is _MISSING or raw is None:
        return _MISSING
    return json.loads(raw)


def unpack_openai_item(item):
    """
    Reconstruct an OpenAI batch item from a packed representation.

    This mirrors `pack_openai_item` and intentionally mutates `item` using `pop`.

    :param item: Packed item with keys produced by `pack_openai_item`.
    :return: Reconstructed OpenAI batch item.
    """
    custom_id = item.pop("custom_id")
    method = item.pop("method")
    url = item.pop("url")

    body_model = item.pop("body.model")
    body_instructions = item.pop("body.instructions")
    body_input = _json_loads_if_not_none(item.pop("body.input"))
    body_tools = _json_loads_if_not_none(item.pop("body.tools"))
    body_text_format = _pop_unpacked_json_or_missing(item, "body.text.format")

    # needs to be a dict
    body_rest = _json_loads_if_not_none(item.pop("body.rest")) or {}
    item_rest = _json_loads_if_not_none(item.pop("item.rest")) or {}

    body = {
        "model": body_model,
        "instructions": body_instructions,
        "input": body_input,
        "tools": body_tools,
        **body_rest,
    }

    if body_text_format is not _MISSING:
        body["text"] = {"format": body_text_format}

    unpacked = {
        "custom_id": custom_id,
        "method": method,
        "url": url,
        "body": body,
        **item_rest,
    }

    return unpacked


if __name__ == "__main__":
    # parent_dir = Path(".pllm/example/stress_test/batch-in")
    # parent_dir = Path(".pllm/example/batch/batch-in")
    parent_dirs = [
        Path(".pllm/example/stress_test/batch-in"),
        Path(".pllm/example/batch/batch-in"),
        Path(".pllm/simplest-tool/batch-in"),
    ]

    all_dfs = []
    for parent_dir in parent_dirs:
        for fname in os.listdir(parent_dir):
            if fname.endswith(".jsonl"):
                # df = pl.read_ndjson(Path(f".pllm/example/stress_test/batch-in/{fname}"))
                # print(df)

                collector = []
                with open(parent_dir / fname, "r", encoding="utf-8") as f:
                    for line in f:
                        obj = json.loads(line)

                        if "body" not in obj:
                            # characteristic of openai items
                            continue
                        collector.append(pack_openai_item(obj))
                if collector:
                    df = pl_json_normalize(collector)
                    all_dfs.append(df)
                    # print(df)

    largest_df = pl.concat(all_dfs, how="diagonal_relaxed")
    pass
