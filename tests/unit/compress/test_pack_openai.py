from copy import deepcopy

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from parallem.core.compress._pack_openai_batch import (
    compress_openai_input_batches,
    pack_openai_item,
    unpack_openai_item,
)

pytestmark = pytest.mark.skip(reason="implementation no longer used")


def test_round_trip_absent_text_format_uses_none_and_stays_absent():
    item = {
        "custom_id": "req_1",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": "gpt-4.1-mini",
            "instructions": "Be concise",
            "input": [{"role": "user", "content": "Hi"}],
            "tools": [],
        },
    }

    packed = pack_openai_item(deepcopy(item))
    assert packed["body.text.format"] is None

    unpacked = unpack_openai_item(deepcopy(packed))
    assert "text" not in unpacked["body"]
    assert unpacked == item


def test_round_trip_present_null_text_format_uses_json_null():
    item = {
        "custom_id": "req_2",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": "gpt-4.1-mini",
            "instructions": "Be concise",
            "input": [{"role": "user", "content": "Hi"}],
            "tools": [],
            "text": {"format": None},
        },
    }

    packed = pack_openai_item(deepcopy(item))
    assert packed["body.text.format"] == "null"

    unpacked = unpack_openai_item(deepcopy(packed))
    assert unpacked["body"]["text"]["format"] is None
    assert unpacked == item


def test_round_trip_preserves_body_and_item_rest_fields():
    item = {
        "custom_id": "req_3",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": "gpt-4.1-mini",
            "instructions": "Be concise",
            "input": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "name": "search"}],
            "metadata": {"source": "test"},
            "reasoning": {"effort": "low"},
        },
        "priority": None,
        "extra": {"nested": [1, 2, 3]},
    }

    packed = pack_openai_item(deepcopy(item))
    unpacked = unpack_openai_item(deepcopy(packed))

    assert unpacked == item


def test_compress_openai_input_batches_packs_nested_fields():
    items = [
        {
            "custom_id": "req_1",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": "gpt-4.1-mini",
                "instructions": "Be concise",
                "input": [{"role": "user", "content": "Hi"}],
                "tools": [],
                "text": {"format": {"type": "json_schema", "strict": True}},
            },
        }
    ]

    df = compress_openai_input_batches(items)

    assert df.schema["body.model"] == pl.Utf8
    assert df.select("body.input").to_series().to_list() == [
        '[{"role": "user", "content": "Hi"}]'
    ]
    assert df.select("body.text.format").to_series().to_list() == [
        '{"type": "json_schema", "strict": true}'
    ]


def test_compress_openai_input_batches_matches_expected_schema():
    items = [
        {
            "custom_id": "req_1",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": "gpt-4o-mini",
                "instructions": "Be concise",
                "input": [{"role": "user", "content": "Hi"}],
                "tools": [],
            },
        }
    ]

    actual = compress_openai_input_batches(items)
    expected = pl.DataFrame(
        {
            "custom_id": ["req_1"],
            "method": ["POST"],
            "url": ["/v1/responses"],
            "body.model": ["gpt-4o-mini"],
            "body.instructions": ["Be concise"],
            "body.input": ['[{"role": "user", "content": "Hi"}]'],
            "body.tools": ["[]"],
            "body.text.format": [None],
            "body.rest": [None],
            "item.rest": [None],
        },
        schema={
            "custom_id": pl.Utf8,
            "method": pl.Utf8,
            "url": pl.Utf8,
            "body.model": pl.Utf8,
            "body.instructions": pl.Utf8,
            "body.input": pl.Utf8,
            "body.tools": pl.Utf8,
            "body.text.format": pl.Utf8,
            "body.rest": pl.Utf8,
            "item.rest": pl.Utf8,
        },
    )

    assert_frame_equal(actual, expected)


def test_compress_openai_input_batches_preserves_custom_ids():
    items = [
        {
            "custom_id": "req_1",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": "gpt-4o-mini",
                "instructions": "Be concise",
                "input": [{"role": "user", "content": "Hi"}],
                "tools": [],
            },
        },
        {
            "custom_id": "req_2",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": "gpt-4o-mini",
                "instructions": "Be concise",
                "input": [{"role": "user", "content": "Hello"}],
                "tools": [],
            },
        },
    ]

    df = compress_openai_input_batches(items)

    assert sorted(df["custom_id"].to_list()) == ["req_1", "req_2"]
