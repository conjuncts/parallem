"""
Unit tests for OpenAI provider classes

Tests the OpenAI provider functionality including:
- SyncOpenAIProvider and ConcurrentOpenAIProvider
- Document formatting for OpenAI API
- Provider type handling
- Integration with backends
- Call ID generation and submission
"""

import pytest
from parallem.provider.openai.sdk import (
    SyncOpenAIProvider,
    ConcurrentOpenAIProvider,
)
from parallem.testing.simple_backend import MockSyncBackend, MockConcurrentBackend
from parallem.core.response import PendingLLMResponse, ReadyLLMResponse
from parallem.types import LLMIdentity
from parallem.testing.simple_mock import MockOpenAIClient, MockConcurrentOpenAIClient
from PIL import Image


pytest.skip("Not very informative", allow_module_level=True)


def _fix_docs_for_openai(*args, **kwargs):
    return SyncOpenAIProvider._fix_docs_for_openai(None, *args, **kwargs)


class TestDocumentFixing:
    """Test document formatting` for OpenAI API"""

    def test_fix_docs_single_string(self):
        """Test fixing a single string document"""
        result = _fix_docs_for_openai(["Hello world"])

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello world"

    def test_fix_docs_list_of_strings(self):
        """Test fixing a list of string documents"""
        docs = ["First message", "Second message"]
        result = _fix_docs_for_openai(docs)

        assert isinstance(result, list)
        assert len(result) == 2

        for i, doc in enumerate(docs):
            assert result[i]["role"] == "user"
            assert result[i]["content"] == doc

    def test_fix_docs_empty_list(self):
        """Test fixing an empty list"""
        result = _fix_docs_for_openai([])

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.skip("Images not yet supported")
    def test_fix_docs_mixed_types(self):
        """Test fixing mixed document types"""
        # Create a test image
        img = Image.new("RGB", (10, 10), color="red")
        docs = ["Text message", img]

        result = _fix_docs_for_openai(docs)

        assert isinstance(result, list)
        assert len(result) == 2

        # First should be text
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Text message"

        # Second should handle image (implementation dependent)
        assert result[1]["role"] == "user"
        # Image handling may vary by implementation

    def test_fix_docs_preserves_structure(self):
        """Test that document fixing preserves message structure"""
        docs = ["Message 1", "Message 2", "Message 3"]
        result = _fix_docs_for_openai(docs)

        # Should maintain order
        for i, original_doc in enumerate(docs):
            assert result[i]["content"] == original_doc
            assert result[i]["role"] == "user"


class TestSyncOpenAIProvider:
    """Test SyncOpenAIProvider functionality with realistic scenarios"""

    def test_successful_query_execution(self, generic_call_id):
        """Test that a query is properly executed and returns expected response"""
        mock_client = MockOpenAIClient()
        mock_client.set_default("The capital of France is Paris.")
        mock_backend = MockSyncBackend()

        provider = SyncOpenAIProvider(client=mock_client, backend=mock_backend)

        result = provider.submit_query_to_provider(
            instructions="Answer the question accurately.",
            documents=["What is the capital of France?"],
            call_id=generic_call_id,
        )

        # Should return ReadyLLMResponse with the actual response
        assert isinstance(result, ReadyLLMResponse)
        assert result._value == "The capital of France is Paris."

        # Verify the client was called with correct parameters
        assert len(mock_client.calls) == 1
        call_data = mock_client.calls[0]
        assert call_data["instructions"] == "Answer the question accurately."
        assert len(call_data["input"]) == 1
        assert call_data["input"][0]["content"] == "What is the capital of France?"

    def test_cached_response_retrieval(self, generic_call_id):
        """Test that cached responses are returned without calling the client again"""
        mock_client = MockOpenAIClient()
        mock_client.set_default("Cached response")
        mock_backend = MockSyncBackend()

        # Pre-populate the cache
        mock_backend.stored_responses[mock_backend._get_cache_key(generic_call_id)] = (
            "Cached response",
            "cached_resp_123",
            {"cached": True},
        )

        provider = SyncOpenAIProvider(client=mock_client, backend=mock_backend)

        result = provider.submit_query_to_provider(
            instructions="Test instructions",
            documents=["Test document"],
            call_id=generic_call_id,
        )

        # Should return cached response
        assert isinstance(result, ReadyLLMResponse)
        assert result._value == "Cached response"

        # Client should not have been called since response was cached
        assert len(mock_client.calls) == 0

    def test_llm_identity_parameter_usage(self, generic_call_id):
        """Test that LLM identity is properly passed to the client"""
        mock_client = MockOpenAIClient()
        mock_client.set_default("Model-specific response")
        mock_backend = MockSyncBackend()

        provider = SyncOpenAIProvider(client=mock_client, backend=mock_backend)
        llm_identity = LLMIdentity("gpt-4")

        result = provider.submit_query_to_provider(
            instructions="Use the specified model",
            documents=["Test input"],
            call_id=generic_call_id,
            llm=llm_identity,
        )

        assert isinstance(result, ReadyLLMResponse)
        assert result._value == "Model-specific response"

        # Verify correct model was used
        assert len(mock_client.calls) == 1
        assert mock_client.calls[0]["model"] == "gpt-4"

    def test_document_processing_and_formatting(self, generic_call_id):
        """Test that multiple documents are properly formatted for OpenAI API"""
        mock_client = MockOpenAIClient()
        mock_client.set_default("Processed multiple documents")
        mock_backend = MockSyncBackend()

        provider = SyncOpenAIProvider(client=mock_client, backend=mock_backend)

        documents = [
            "First document content",
            "Second document content",
            "Third document content",
        ]

        result = provider.submit_query_to_provider(
            instructions="Process these documents",
            documents=documents,
            call_id=generic_call_id,
        )

        assert isinstance(result, ReadyLLMResponse)

        # Verify documents were properly formatted
        assert len(mock_client.calls) == 1
        input_messages = mock_client.calls[0]["input"]
        assert len(input_messages) == 3

        for i, doc in enumerate(documents):
            assert input_messages[i]["role"] == "user"
            assert input_messages[i]["content"] == doc

    def test_additional_kwargs_passthrough(self, generic_call_id):
        """Test that additional parameters are passed through to the client"""
        mock_client = MockOpenAIClient()
        mock_client.set_default("Response with custom params")
        mock_backend = MockSyncBackend()

        provider = SyncOpenAIProvider(client=mock_client, backend=mock_backend)

        result = provider.submit_query_to_provider(
            instructions="Test with custom parameters",
            documents=["Test input"],
            call_id=generic_call_id,
            temperature=0.7,
            max_tokens=150,
            top_p=0.9,
        )

        assert isinstance(result, ReadyLLMResponse)

        # Verify additional parameters were passed through
        call_data = mock_client.calls[0]
        assert call_data["kwargs"]["temperature"] == 0.7
        assert call_data["kwargs"]["max_tokens"] == 150
        assert call_data["kwargs"]["top_p"] == 0.9

    def test_error_handling_from_client(self, generic_call_id):
        """Test that client errors are properly propagated"""
        mock_client = MockOpenAIClient()
        mock_backend = MockSyncBackend()

        # Make the client raise an exception
        def failing_create(*args, **kwargs):
            raise Exception("OpenAI API Error: Rate limit exceeded")

        mock_client.responses.create = failing_create
        provider = SyncOpenAIProvider(client=mock_client, backend=mock_backend)

        with pytest.raises(Exception, match="OpenAI API Error: Rate limit exceeded"):
            provider.submit_query_to_provider(
                instructions="This will fail",
                documents=["Test"],
                call_id=generic_call_id,
            )


class TestConcurrentOpenAIProvider:
    """Test ConcurrentOpenAIProvider functionality with realistic concurrent scenarios"""

    @pytest.mark.asyncio
    async def test_successful_concurrent_query_execution(self, generic_call_id):
        """Test that a concurrent query is properly executed and can be resolved"""
        mock_client = MockConcurrentOpenAIClient()
        mock_client.set_default("Concurrent response: The capital of Italy is Rome.")
        mock_backend = MockConcurrentBackend()

        provider = ConcurrentOpenAIProvider(client=mock_client, backend=mock_backend)

        result = provider.submit_query_to_provider(
            instructions="Answer the geographical question.",
            documents=["What is the capital of Italy?", "Answer in a full sentence."],
            call_id=generic_call_id,
        )

        # Should return PendingLLMResponse
        assert isinstance(result, PendingLLMResponse)
        assert result.call_id == generic_call_id
        assert result._backend == mock_backend

        # Resolve the pending response
        resolved_value = await mock_backend.resolve_call(generic_call_id)
        assert resolved_value == "Concurrent response: The capital of Italy is Rome."

    @pytest.mark.asyncio
    async def test_concurrent_llm_identity_usage(self, generic_call_id):
        """Test that LLM identity is properly used in concurrent calls"""
        mock_client = MockConcurrentOpenAIClient()
        mock_client.set_default("GPT-3.5 specific response")
        mock_backend = MockConcurrentBackend()

        provider = ConcurrentOpenAIProvider(client=mock_client, backend=mock_backend)
        llm_identity = LLMIdentity("gpt-3.5-turbo")

        result = provider.submit_query_to_provider(
            instructions="Use GPT-3.5 for this task",
            documents=["Test concurrent input"],
            call_id=generic_call_id,
            llm=llm_identity,
        )

        assert isinstance(result, PendingLLMResponse)

        # Execute the coroutine to verify the model was used
        resolved_value = await mock_backend.resolve_call(generic_call_id)
        assert resolved_value == "GPT-3.5 specific response"


class TestProviderIntegration:
    """Test provider integration and cross-cutting concerns"""

    @pytest.mark.asyncio
    async def test_sync_vs_concurrent_behavior(self, generic_call_id):
        """Test that sync and concurrent providers produce equivalent results"""
        mock_sync_client = MockOpenAIClient()
        mock_concurrent_client = MockConcurrentOpenAIClient()

        # Set same response for both
        response_text = "Both providers should return this"
        mock_sync_client.set_default(response_text)
        mock_concurrent_client.set_default(response_text)

        sync_backend = MockSyncBackend()
        concurrent_backend = MockConcurrentBackend()

        sync_provider = SyncOpenAIProvider(client=mock_sync_client, backend=sync_backend)
        concurrent_provider = ConcurrentOpenAIProvider(
            client=mock_concurrent_client, backend=concurrent_backend
        )

        # Same input for both
        instructions = "Test instruction consistency"
        documents = ["Document for consistency test"]

        # Test sync provider
        sync_result = sync_provider.submit_query_to_provider(
            instructions=instructions, documents=documents, call_id=generic_call_id
        )

        # Test concurrent provider
        concurrent_call_id = generic_call_id.copy()
        concurrent_call_id["seq_id"] = 2  # Different seq_id to avoid cache collision

        concurrent_result = concurrent_provider.submit_query_to_provider(
            instructions=instructions, documents=documents, call_id=concurrent_call_id
        )

        # Sync should return ReadyLLMResponse
        assert isinstance(sync_result, ReadyLLMResponse)
        assert sync_result._value == response_text

        # Concurrent should return PendingLLMResponse that resolves to same value
        assert isinstance(concurrent_result, PendingLLMResponse)
        resolved_concurrent_value = await concurrent_backend.resolve_call(concurrent_call_id)
        assert resolved_concurrent_value == response_text


class TestProviderErrorScenarios:
    """Test error scenarios and edge cases with realistic behavior"""

    def test_none_instructions_handling(self, generic_call_id):
        """Test that providers handle None instructions gracefully"""
        mock_client = MockOpenAIClient()
        mock_client.set_default("Response with no instructions/documents")
        mock_backend = MockSyncBackend()

        provider = SyncOpenAIProvider(client=mock_client, backend=mock_backend)

        result = provider.submit_query_to_provider(
            instructions=None, documents=[], call_id=generic_call_id
        )

        assert isinstance(result, ReadyLLMResponse)
        assert result._value == "Response with no instructions/documents"

        # Verify None instructions were passed through
        assert len(mock_client.calls) == 1
        assert mock_client.calls[0]["instructions"] is None
        assert mock_client.calls[0]["input"] == []

    def test_client_api_error_propagation(self, generic_call_id):
        """Test that API errors from the client are properly propagated"""
        mock_client = MockOpenAIClient()
        mock_backend = MockSyncBackend()

        # Configure client to raise an exception
        def failing_create(*args, **kwargs):
            raise Exception("API Rate limit exceeded")

        mock_client.responses.create = failing_create
        provider = SyncOpenAIProvider(client=mock_client, backend=mock_backend)

        with pytest.raises(Exception, match="API Rate limit exceeded"):
            provider.submit_query_to_provider(
                instructions="This will fail",
                documents=["Test document"],
                call_id=generic_call_id,
            )

    @pytest.mark.asyncio
    async def test_concurrent_client_error_propagation(self, generic_call_id):
        """Test that concurrent API errors are properly propagated"""
        mock_client = MockConcurrentOpenAIClient()
        mock_backend = MockConcurrentBackend()

        # Configure concurrent client to raise an exception
        async def failing_concurrent_create(*args, **kwargs):
            raise Exception("Concurrent API timeout")

        mock_client.responses.create = failing_concurrent_create
        provider = ConcurrentOpenAIProvider(client=mock_client, backend=mock_backend)

        result = provider.submit_query_to_provider(
            instructions="This will fail concurrent",
            documents=["Test document"],
            call_id=generic_call_id,
        )

        assert isinstance(result, PendingLLMResponse)

        # The error should be raised when we try to resolve
        with pytest.raises(Exception, match="Concurrent API timeout"):
            await mock_backend.resolve_call(generic_call_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
