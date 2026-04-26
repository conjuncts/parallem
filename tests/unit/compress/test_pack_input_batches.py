from copy import deepcopy

from parallem.core.compress.pack_input_batches import pack_openai_item
from parallem.core.compress.pack_input_batches import unpack_openai_item


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
