"""
Unit tests for cast_document_to_bytes function

Tests the document casting functionality including:
- Converting various document types to bytes
- Proper type identification
- Handling of metadata/extra fields
"""

import pytest
from PIL import Image
from io import BytesIO
from unittest.mock import Mock

from pipelinellm.core.cast.doc_to_str import (
    cast_bytes_to_document,
    cast_document_to_bytes,
)
from pipelinellm.types import (
    FunctionCallRequest,
    FunctionCallOutput,
    FunctionCall,
    LLMResponse,
    CallIdentifier,
)


class TestCastDocumentToBytes:
    """Test the cast_document_to_bytes function"""

    @pytest.fixture
    def sample_call_id(self) -> CallIdentifier:
        """Create a sample call identifier for testing"""
        return {
            "agent_name": "test_agent",
            "doc_hash": "test_hash",
            "seq_id": 42,
            "session_id": 1,
            "meta": {"provider_type": "openai", "tag": None},
        }

    def test_cast_string(self):
        """Test casting a simple string document"""
        doc = "Hello, world!"
        doc_bytes, doc_type, doc_extra = cast_document_to_bytes(doc)

        assert doc_bytes == b"Hello, world!"
        assert doc_type == "text"
        assert doc_extra is None

    def test_cast_empty_string(self):
        """Test casting an empty string"""
        doc = ""
        doc_bytes, doc_type, doc_extra = cast_document_to_bytes(doc)

        assert doc_bytes == b""
        assert doc_type == "text"
        assert doc_extra is None

    def test_cast_unicode_string(self):
        """Test casting a string with unicode characters"""
        doc = "Hello 世界! 🌍"
        doc_bytes, doc_type, doc_extra = cast_document_to_bytes(doc)

        assert doc_bytes == "Hello 世界! 🌍".encode("utf-8")
        assert doc_type == "text"
        assert doc_extra is None

    def test_cast_function_call_request(self, sample_call_id):
        """Test casting a FunctionCallRequest"""
        func_call = FunctionCall(
            name="test_function",
            arguments='{"arg": "value"}',
            call_id="func_123",
        )
        doc = FunctionCallRequest(
            text_content="Calling test function",
            calls=[func_call],
            call_id=sample_call_id,
        )

        doc_bytes, doc_type, doc_extra = cast_document_to_bytes(doc)

        # Should encode the call_id as serial ID
        expected = "test_agent:42:1".encode("utf-8")
        assert doc_bytes == expected
        assert doc_type == "function_call"
        assert doc_extra is None

    def test_cast_function_call_output(self):
        """Test casting a FunctionCallOutput"""
        doc = FunctionCallOutput(
            content="Function output result",
            call_id="func_123",
            name="test_function",
        )

        doc_bytes, doc_type, doc_extra = cast_document_to_bytes(doc)

        assert doc_bytes == b"Function output result"
        assert doc_type == "function_call_output"
        assert doc_extra == '{"c":"func_123","n":"test_function"}'

    def test_cast_tuple(self):
        """Test casting a tuple (role, content)"""
        doc = ("system", "You are a helpful assistant")

        doc_bytes, doc_type, doc_extra = cast_document_to_bytes(doc)

        assert doc_bytes == b"You are a helpful assistant"
        assert doc_type == "text"
        assert doc_extra == "system"

    def test_cast_tuple_with_unicode(self):
        """Test casting a tuple with unicode content"""
        doc = ("user", "こんにちは")

        doc_bytes, doc_type, doc_extra = cast_document_to_bytes(doc)

        assert doc_bytes == "こんにちは".encode("utf-8")
        assert doc_type == "text"
        assert doc_extra == "user"

    def test_cast_image(self):
        """Test casting a PIL Image"""
        # Create a simple test image
        img = Image.new("RGB", (10, 10), color="red")

        doc_bytes, doc_type, doc_extra = cast_document_to_bytes(img)

        assert isinstance(doc_bytes, bytes)
        assert len(doc_bytes) > 0
        assert doc_type == "image"
        assert doc_extra is None

        # Verify the bytes can be decoded back to an image
        buffered = BytesIO(doc_bytes)
        restored_img = Image.open(buffered)
        assert restored_img.size == (10, 10)

    def test_cast_llm_response(self, sample_call_id):
        """Test casting an LLMResponse"""
        doc = LLMResponse("Response content", call_id=sample_call_id)

        doc_bytes, doc_type, doc_extra = cast_document_to_bytes(doc)

        # Should encode the call_id as serial ID
        expected = "test_agent:42:1".encode("utf-8")
        assert doc_bytes == expected
        assert doc_type == "llm_response"
        assert doc_extra is None

    def test_cast_unknown_type(self):
        """Test that unknown types raise NotImplementedError"""
        doc = {"some": "dict"}  # Dictionary is not a supported type

        with pytest.raises(NotImplementedError) as exc_info:
            cast_document_to_bytes(doc)

        assert "Unknown document type" in str(exc_info.value)
        assert "dict" in str(exc_info.value)


class TestCastBytesToDocument:
    """Tests for cast_bytes_to_document (reverse of cast_document_to_bytes)."""

    def test_roundtrip_string(self):
        doc = "Hello, world!"
        byt, dtype, dextra = cast_document_to_bytes(doc)
        result = cast_bytes_to_document(byt, dtype, dextra)
        assert result == doc

    def test_roundtrip_tuple(self):
        doc = ("system", "You are helpful")
        byt, dtype, dextra = cast_document_to_bytes(doc)
        result = cast_bytes_to_document(byt, dtype, dextra)
        assert result == doc

    def test_roundtrip_image(self):
        img = Image.new("RGB", (8, 8), color="blue")
        byt, dtype, dextra = cast_document_to_bytes(img)
        result = cast_bytes_to_document(byt, dtype, dextra)
        assert isinstance(result, Image.Image)
        assert result.size == (8, 8)

    def test_function_call_output(self):
        """FunctionCallOutput is reconstructed."""
        doc = FunctionCallOutput(content="result", call_id="cid_99", name="my_fn")
        byt, dtype, dextra = cast_document_to_bytes(doc)
        result = cast_bytes_to_document(byt, dtype, dextra)
        assert isinstance(result, FunctionCallOutput)
        assert result.content == "result"
        assert result.call_id == "cid_99"
        assert result.name == "my_fn"

    def test_function_call_stub(self):
        """FunctionCallRequest round-trips to a stub with only call_id."""
        call_id = {
            "agent_name": "a",
            "doc_hash": "h",
            "seq_id": 1,
            "session_id": 0,
            "meta": None,
        }
        doc = FunctionCallRequest(text_content="t", calls=[], call_id=call_id)
        byt, dtype, dextra = cast_document_to_bytes(doc)
        retriever = Mock()
        full_call_id = {
            "agent_name": "a",
            "doc_hash": "h",
            "seq_id": 1,
            "session_id": 0,
        }
        retriever.populate_call_id.return_value = full_call_id
        retrieved = Mock()
        retrieved.text = "hydrated text"
        retrieved.function_calls = []
        retriever.retrieve.return_value = retrieved

        result = cast_bytes_to_document(byt, dtype, dextra, retriever=retriever)
        assert isinstance(result, FunctionCallRequest)
        assert result.call_id == full_call_id
        assert result.text_content == "hydrated text"
        assert result.calls == []
        retriever.populate_call_id.assert_called_once_with(
            {"agent_name": "a", "seq_id": 1, "session_id": 0}
        )
        retriever.retrieve.assert_called_once_with(full_call_id)

    def test_llm_response_stub(self):
        """LLMResponse round-trips to a stub with only call_id."""
        call_id = {
            "agent_name": "a",
            "doc_hash": "h",
            "seq_id": 1,
            "session_id": 0,
            "meta": None,
        }
        doc = LLMResponse("hello", call_id=call_id)
        byt, dtype, dextra = cast_document_to_bytes(doc)
        retriever = Mock()
        full_call_id = {
            "agent_name": "a",
            "doc_hash": "h",
            "seq_id": 1,
            "session_id": 0,
        }
        retriever.populate_call_id.return_value = full_call_id

        result = cast_bytes_to_document(byt, dtype, dextra, retriever=retriever)
        assert isinstance(result, LLMResponse)
        assert result.call_id == full_call_id
        assert result.value is None
        retriever.populate_call_id.assert_called_once_with(
            {"agent_name": "a", "seq_id": 1, "session_id": 0}
        )

    def test_unknown_type_raises(self):
        with pytest.raises(NotImplementedError):
            cast_bytes_to_document(b"x", "unknown_type", None)
