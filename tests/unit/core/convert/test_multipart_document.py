import tempfile
from pathlib import Path
import pytest
from PIL import Image

from parallem.types import MultipartDocument, LLMIdentity
from parallem.core.convert.completions import from_chat_completion, to_chat_completion
from parallem.core.hash import compute_hash
from parallem.core.file_manager import FileManager
from parallem.core.datastore.input_storage import InputStorage


def _sample_call_id(session_id=1, seq_id=2):
    return {
        "agent_name": "test_agent",
        "doc_hash": "test_hash",
        "seq_id": seq_id,
        "session_id": session_id,
        "meta": {"provider_type": "openai", "tag": None},
    }


def test_multipart_document_creation():
    # Test valid creation
    img = Image.new("RGB", (4, 4), color="red")
    doc = MultipartDocument(parts=["Hello", img])
    assert doc.parts == ["Hello", img]
    assert doc.type == "multipart"

    # Test invalid creation (empty parts)
    with pytest.raises(ValueError, match="MultipartDocument must have at least one part"):
        MultipartDocument(parts=[])


def test_multipart_document_hashing():
    img1 = Image.new("RGB", (4, 4), color="blue")
    img2 = Image.new("RGB", (4, 4), color="green")

    doc1 = MultipartDocument(parts=["text", img1])
    doc2 = MultipartDocument(parts=["text", img2])
    doc3 = MultipartDocument(parts=["different text", img1])
    doc4 = MultipartDocument(parts=["text", img1])

    hash1 = compute_hash(None, [doc1])
    hash2 = compute_hash(None, [doc2])
    hash3 = compute_hash(None, [doc3])
    hash4 = compute_hash(None, [doc4])

    assert hash1 != hash2
    assert hash1 != hash3
    assert hash1 == hash4


def test_multipart_document_conversion():
    img = Image.new("RGB", (4, 4), color="red")
    original = MultipartDocument(parts=["Hello user", img])

    completion_msg = to_chat_completion(original)
    assert completion_msg["role"] == "user"
    assert len(completion_msg["content"]) == 2
    assert completion_msg["content"][0]["type"] == "text"
    assert completion_msg["content"][0]["text"] == "Hello user"
    assert completion_msg["content"][1]["type"] == "image_url"

    restored = from_chat_completion(completion_msg)
    assert isinstance(restored, MultipartDocument)
    assert len(restored.parts) == 2
    assert restored.parts[0] == "Hello user"
    assert isinstance(restored.parts[1], Image.Image)
    assert restored.parts[1].size == (4, 4)


def test_multipart_document_save_input():
    with tempfile.TemporaryDirectory() as temp_dir:
        fm = FileManager(Path(temp_dir))
        storage = InputStorage(fm)

        img = Image.new("RGB", (4, 4), color="red")
        doc = MultipartDocument(parts=["Part 1 text", img])

        call_id = _sample_call_id(session_id=1, seq_id=2)
        storage.store_input(
            call_id,
            params={
                "instructions": "Test Instructions",
                "llm": LLMIdentity("gpt-4o"),
                "strict_documents": [doc],
                "structured_output": None,
                "tools": None,
            },
            hash_by=None,
            salt=None,
            request_kwargs={},
        )
        storage.persist()

        # Retrieve the input and verify
        retrieved = storage.retrieve_input(call_id["doc_hash"], 0)
        assert isinstance(retrieved, MultipartDocument)
        assert len(retrieved.parts) == 2
        assert retrieved.parts[0] == "Part 1 text"
        assert isinstance(retrieved.parts[1], Image.Image)
        assert retrieved.parts[1].size == (4, 4)
