import json
from unittest.mock import Mock

from pipelinellm.provider.anthropic.sdk import BatchAnthropicProvider


def test_decode_batch_content_with_success_line():
    provider = BatchAnthropicProvider(client=Mock())

    content = json.dumps(
        {
            "custom_id": "anth-1",
            "result": {
                "type": "succeeded",
                "message": {
                    "id": "msg_1",
                    "content": [{"type": "text", "text": "Hello from batch"}],
                },
            },
        }
    )

    batch_results = provider.decode_batch_content(content)

    assert len(batch_results) == 1
    assert batch_results[0].status == "ready"
    assert batch_results[0].parsed_responses is not None
    assert len(batch_results[0].parsed_responses) == 1
    parsed = batch_results[0].parsed_responses[0]
    assert parsed.custom_id == "anth-1"
    assert parsed.response_id == "msg_1"
    assert parsed.text == "Hello from batch"


def test_decode_batch_content_with_error_line():
    provider = BatchAnthropicProvider(client=Mock())

    content = json.dumps(
        {
            "custom_id": "anth-err",
            "result": {
                "type": "errored",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Bad request",
                    "status_code": 400,
                },
            },
        }
    )

    batch_results = provider.decode_batch_content(content)

    assert len(batch_results) == 1
    assert batch_results[0].status == "error"
    assert batch_results[0].parsed_responses is not None
    assert len(batch_results[0].parsed_responses) == 1
    parsed = batch_results[0].parsed_responses[0]
    assert parsed.custom_id == "anth-err"
    assert parsed.error_code == 400
    assert parsed.text == "Bad request"


def test_decode_batch_content_with_mixed_lines():
    provider = BatchAnthropicProvider(client=Mock())

    success = {
        "custom_id": "anth-ok",
        "result": {
            "type": "succeeded",
            "message": {
                "id": "msg_ok",
                "content": [{"type": "text", "text": "ok"}],
            },
        },
    }
    expired = {
        "custom_id": "anth-exp",
        "result": {
            "type": "expired",
        },
    }

    content = "\n".join([json.dumps(success), json.dumps(expired)])
    batch_results = provider.decode_batch_content(content)

    assert len(batch_results) == 2

    ready_result = next(result for result in batch_results if result.status == "ready")
    error_result = next(result for result in batch_results if result.status == "error")

    assert ready_result.parsed_responses is not None
    assert ready_result.parsed_responses[0].custom_id == "anth-ok"

    assert error_result.parsed_responses is not None
    assert error_result.parsed_responses[0].custom_id == "anth-exp"
