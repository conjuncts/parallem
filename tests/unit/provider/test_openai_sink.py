import json
import polars as pl

from parallem.provider.openai._sink import openai_metadata_sinker


def test_openai_metadata_sinker_reasoning_before_message():
    """
    Regression test for schema inference bug when reasoning output comes before message output.

    When OpenAI responses include reasoning items before message items in the output array,
    the first item has no 'role' field (None), while subsequent message items have role='assistant'.
    This caused Polars to infer the 'role' column as Null type, then fail when appending strings.

    This test ensures the explicit schema handles mixed None and string values correctly.
    """
    # Sample metadata with reasoning output before message output (the problematic case)
    metadata_json = json.dumps(
        {
            "object": "response",
            "created_at": 1773097304,
            "status": "completed",
            "model": "gpt-5-nano-2025-08-07",
            "output": [
                {
                    "id": "rs_0b3b0c9498caafaf0069af5159206881a3a8f9a54b4c9c991f",
                    "type": "reasoning",
                    "summary": [],
                },
                {
                    "id": "msg_0b3b0c9498caafaf0069af516b237481a3832447d053a759f6",
                    "type": "message",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "annotations": [],
                            "logprobs": [],
                            "text": "other",
                        }
                    ],
                    "role": "assistant",
                },
            ],
            "usage": {
                "input_tokens": 1102,
                "output_tokens": 3386,
                "total_tokens": 4488,
            },
        }
    )

    # This should not raise an error about appending string to Null type
    result = openai_metadata_sinker([({"response_id": "test_123"}, metadata_json)])

    # Verify the structure
    assert "responses" in result
    assert "messages" in result

    messages_df = result["messages"]

    # Check that we have 2 messages (reasoning + message)
    assert len(messages_df) == 2

    # Check that the role column exists and has expected values
    assert "role" in messages_df.columns
    roles = messages_df.select("role").to_series().to_list()
    assert roles[0] is None  # reasoning item has no role
    assert roles[1] == "assistant"  # message item has role

    # Check types
    assert messages_df.select("type").to_series().to_list() == ["reasoning", "message"]

    # Verify the schema is explicitly typed as Utf8 (not Null)
    assert messages_df.schema["role"] == pl.Utf8


def test_openai_metadata_sinker_message_only():
    """
    Test that the fix doesn't break normal cases with only message outputs.
    """
    metadata_json = json.dumps(
        {
            "object": "response",
            "created_at": 1773097304,
            "status": "completed",
            "model": "gpt-4",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "status": "completed",
                    "content": [{"type": "text", "text": "Hello"}],
                    "role": "assistant",
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
    )

    result = openai_metadata_sinker([({"response_id": "test_456"}, metadata_json)])

    messages_df = result["messages"]
    assert len(messages_df) == 1
    assert messages_df.select("role").to_series().to_list() == ["assistant"]
    assert messages_df.schema["role"] == pl.Utf8
