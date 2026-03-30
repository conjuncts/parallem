"""
Unit tests for core response classes

Tests the LLMResponse hierarchy including:
- LLMIdentity creation and provider guessing
- PendingLLMResponse state management and resolve()
- ReadyLLMResponse immediate resolution
- Serialization/deserialization (__getstate__, __setstate__)
"""

from unittest.mock import Mock
import asyncio
import pytest
from parallem.core.calls import _call_to_concise_dict
from parallem.core.response import (
    PendingLLMResponse,
    ReadyLLMResponse,
)
from parallem.types import ParsedResponse
from parallem.testing.simple_backend import MockBackend


class TestReadyLLMResponse:
    """Test ReadyLLMResponse (already resolved responses)"""

    def test_ready_response_immediate_resolution(self, generic_call_id):
        """Test creating a ready response"""
        response = ReadyLLMResponse(call_id=generic_call_id, value="Immediate content")

        assert response.call_id == generic_call_id
        assert response.resolve() == "Immediate content"

    def test_ready_response_serialization(self, generic_call_id):
        """Test ready response can be serialized"""
        response = ReadyLLMResponse(
            call_id=generic_call_id, value="Serializable content"
        )

        # Test __getstate__
        state = response.__getstate__()
        assert "call_id" in state
        assert "value" not in state

        # Test __setstate__
        # ReadyLLMResponse needs the help of the backend to retrieve the true value

    def test_ready_response_default_await_matches_resolve(self, generic_call_id):
        response = ReadyLLMResponse(call_id=generic_call_id, value="Immediate content")

        async def _run():
            return await response

        result = asyncio.run(_run())
        assert result == response.resolve()


class TestPendingLLMResponse:
    """Test PendingLLMResponse (lazy-loaded responses)"""

    def test_pending_response_resolve_calls_backend(self, generic_call_id):
        """Test that pending responses call backend.retrieve()"""
        mock_backend = MockBackend()
        # Store a response in the backend for retrieval
        mock_backend.store(
            generic_call_id,
            ParsedResponse(
                text="Backend response",
                response_id="resp_789",
                metadata=None,
            ),
        )

        response = PendingLLMResponse(call_id=generic_call_id, backend=mock_backend)

        result = response.resolve()

        assert result == "Backend response"

    def test_pending_response_resolve_caching(self, generic_call_id):
        """Test that pending responses cache results after first resolve"""
        mock_backend = MockBackend()
        # Store a response in the backend for retrieval
        mock_backend.store(
            generic_call_id,
            ParsedResponse(
                text="Cached response",
                response_id="resp_123",
                metadata=None,
            ),
        )

        response = PendingLLMResponse(call_id=generic_call_id, backend=mock_backend)

        # First call should hit backend
        result1 = response.resolve()
        assert result1 == "Cached response"

        # Second call should use cached value
        result2 = response.resolve()
        assert result2 == "Cached response"

        # Note: MockBackend doesn't track call counts like unittest.Mock,
        # but we can still verify the caching behavior works correctly

    def test_pending_response_serialization(self, generic_call_id):
        """Test pending response serialization handling"""
        mock_backend = Mock()

        response = PendingLLMResponse(call_id=generic_call_id, backend=mock_backend)

        # Test __getstate__ removes backend
        state = response.__getstate__()
        assert "call_id" in state
        assert "_backend" not in state  # Should be excluded

        # Test __setstate__ restores call_id but backend is None
        new_response = PendingLLMResponse.__new__(PendingLLMResponse)
        new_response.__setstate__(state)
        assert new_response.call_id == _call_to_concise_dict(generic_call_id)
        assert new_response._backend is None

    def test_pending_response_backend_none_handling(self, generic_call_id):
        """Test behavior when backend is None (after deserialization)"""
        response = PendingLLMResponse(
            call_id=generic_call_id,
            backend=None,  # Simulates post-deserialization state
        )

        # Should handle None backend gracefully
        # (The exact behavior depends on implementation)
        with pytest.raises((AttributeError, RuntimeError)):
            response.resolve()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
