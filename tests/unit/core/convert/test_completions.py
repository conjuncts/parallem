from PIL import Image

from parallem.core.convert.completions import from_chat_completion, to_chat_completion
from parallem.types import FileInput, FunctionCall, FunctionCallOutput, FunctionCallRequest


def _sample_call_id():
    return {
        "agent_name": "test_agent",
        "doc_hash": "test_hash",
        "seq_id": 7,
        "session_id": 3,
        "meta": {"provider_type": "openai", "tag": None},
    }


def test_roundtrip_user_text():
    original = "hello completions"

    completion_msg = to_chat_completion(original)
    restored = from_chat_completion(completion_msg)

    assert restored == original


def test_roundtrip_user_image():
    original = Image.new("RGB", (4, 4), color="green")

    completion_msg = to_chat_completion(original)
    restored = from_chat_completion(completion_msg)

    assert isinstance(restored, Image.Image)
    assert restored.size == (4, 4)


def test_roundtrip_user_file_input_with_content():
    original = FileInput(
        mime_type="text/plain",
        filename="notes.txt",
        file_content=b"hello file",
    )

    completion_msg = to_chat_completion(original)
    restored = from_chat_completion(completion_msg)

    assert isinstance(restored, FileInput)
    assert restored.mime_type == "text/plain"
    assert restored.filename == "notes.txt"
    assert restored.file_content == b"hello file"
    assert restored.file_url is None


def test_roundtrip_assistant_function_call_request():
    original = FunctionCallRequest(
        text_content="thinking about tools",
        calls=[
            FunctionCall(
                name="sum_numbers",
                arguments={"a": 1, "b": 2},
                fcall_id="call_123",
            )
        ],
        call_id=_sample_call_id(),
    )

    completion_msg = to_chat_completion(original)
    restored = from_chat_completion(completion_msg)

    assert isinstance(restored, FunctionCallRequest)
    assert restored.text_content == "thinking about tools"
    assert len(restored.calls) == 1
    assert restored.calls[0].name == "sum_numbers"
    assert restored.calls[0].fcall_id == "call_123"
    assert restored.calls[0].args == {"a": 1, "b": 2}


def test_roundtrip_tool_function_call_output():
    original = FunctionCallOutput(
        content="tool result",
        fcall_id="call_456",
        name="sum_numbers",
    )

    completion_msg = to_chat_completion(original)
    restored = from_chat_completion(completion_msg)

    assert isinstance(restored, FunctionCallOutput)
    assert restored.content == "tool result"
    assert restored.fcall_id == "call_456"
    assert restored.name == ""
