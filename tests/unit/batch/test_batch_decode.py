from tests.unit.batch.data import total_batch_result
import json
from unittest.mock import Mock
from pipelinellm.provider.openai.sdk import BatchOpenAIProvider


def test_decode_batch_content_with_tool_call():
    """Test decode_batch_content with a batch result containing a tool call"""
    # Create a mock OpenAI client
    mock_client = Mock()
    provider = BatchOpenAIProvider(client=mock_client)

    # Extract a single line with count_files function call (custom_id: -0-2-2)
    lines = total_batch_result.strip().split("\n")
    count_files_result = lines[2]  # Third line contains count_files call

    batch_results = provider.decode_batch_content(count_files_result)

    # Should return a single BatchResult with status "ready"
    assert len(batch_results) == 1
    batch_result = batch_results[0]
    assert batch_result.status == "ready"
    assert batch_result.raw_output == count_files_result

    # Should have one parsed response
    assert batch_result.parsed_responses is not None
    assert len(batch_result.parsed_responses) == 1
    parsed_response = batch_result.parsed_responses[0]

    # Verify the custom_id
    assert parsed_response.custom_id == "-0-2-2"

    # Verify the response_id
    assert (
        parsed_response.response_id
        == "resp_02c409ef6120190c0069ac3a88a5788193bcc0a0cfdc109b3d"
    )

    # Verify the function calls
    assert parsed_response.function_calls is not None
    assert len(parsed_response.function_calls) == 1
    function_call = parsed_response.function_calls[0]

    # Verify function call details
    assert function_call.name == "count_files"
    assert function_call.call_id == "call_vjTQNlYFTWhxRjwVglvs6Ih7"
    assert function_call.args == {"directory": "~/examples"}

    # Verify metadata includes usage information
    assert parsed_response.metadata is not None
    assert "usage" in parsed_response.metadata
    assert parsed_response.metadata["usage"]["total_tokens"] == 264


def test_decode_batch_content_with_multiple_lines():
    """Test decode_batch_content with multiple batch results in JSONL format"""
    mock_client = Mock()
    provider = BatchOpenAIProvider(client=mock_client)

    # Use the full total_batch_result which contains 5 lines
    batch_results = provider.decode_batch_content(total_batch_result)

    # Should return a single BatchResult with status "ready" containing all responses
    assert len(batch_results) == 1
    batch_result = batch_results[0]
    assert batch_result.status == "ready"

    # Should have five parsed responses
    assert batch_result.parsed_responses is not None
    assert len(batch_result.parsed_responses) == 5

    # Verify first response (custom_id: -0-0-0, message response)
    first_response = batch_result.parsed_responses[0]
    assert first_response.custom_id == "-0-0-0"
    assert (
        first_response.function_calls is None or len(first_response.function_calls) == 0
    )

    # Verify third response (custom_id: -0-2-2, count_files tool call)
    third_response = batch_result.parsed_responses[2]
    assert third_response.custom_id == "-0-2-2"
    assert len(third_response.function_calls) == 1
    assert third_response.function_calls[0].name == "count_files"


def test_decode_batch_content_with_error():
    """Test decode_batch_content with an error response"""
    mock_client = Mock()
    provider = BatchOpenAIProvider(client=mock_client)

    # Create an error response (status_code != 200)
    first_batch_result = total_batch_result.strip().split("\n")[0]
    error_data = json.loads(first_batch_result)
    error_data["response"]["status_code"] = 500
    error_data["custom_id"] = "error-test-id"
    error_line = json.dumps(error_data)

    batch_results = provider.decode_batch_content(error_line)

    # Should return a single BatchResult with status "error"
    assert len(batch_results) == 1
    batch_result = batch_results[0]
    assert batch_result.status == "error"

    # Should have one error response
    assert batch_result.parsed_responses is not None
    assert len(batch_result.parsed_responses) == 1
    error_response = batch_result.parsed_responses[0]

    # Verify the custom_id and error_code
    assert error_response.custom_id == "error-test-id"
    assert error_response.error_code == 500


def test_decode_batch_content_with_mixed_success_and_error():
    """Test decode_batch_content with both successful and error responses"""
    mock_client = Mock()
    provider = BatchOpenAIProvider(client=mock_client)
    lines = total_batch_result.strip().split("\n")
    success_line = lines[2]  # count_files call
    error_data = json.loads(lines[0])
    error_data["response"]["status_code"] = 400
    error_data["custom_id"] = "error-id"
    error_line = json.dumps(error_data)

    # Combine success and error
    mixed_content = f"{success_line}\n{error_line}"

    batch_results = provider.decode_batch_content(mixed_content)

    # Should return two BatchResults: one ready, one error
    assert len(batch_results) == 2

    # Find the ready and error results
    ready_result = next(br for br in batch_results if br.status == "ready")
    error_result = next(br for br in batch_results if br.status == "error")

    # Verify ready result
    assert ready_result.parsed_responses is not None
    assert len(ready_result.parsed_responses) == 1
    assert ready_result.parsed_responses[0].custom_id == "-0-2-2"

    # Verify error result
    assert error_result.parsed_responses is not None
    assert len(error_result.parsed_responses) == 1
    assert error_result.parsed_responses[0].custom_id == "error-id"
    assert error_result.parsed_responses[0].error_code == 400


def test_cancel_batch_calls_openai_batches_cancel():
    mock_client = Mock()
    provider = BatchOpenAIProvider(client=mock_client)

    provider.cancel_batch("batch_123", provider_type="openai")

    mock_client.batches.cancel.assert_called_once_with("batch_123")
