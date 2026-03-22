from pathlib import Path
import tempfile
from unittest.mock import Mock
import pytest
from pipelinellm.core.response import ReadyLLMResponse
from pipelinellm.types import CallIdentifier, LLMIdentity


@pytest.fixture
def generic_call_id() -> CallIdentifier:
    """Fixture to create mock call identifiers for testing"""
    return {
        "agent_name": "test_agent",
        "doc_hash": "test_hash_1237",
        "seq_id": 1,
        "session_id": 1,
        "meta": {"provider_type": "openai", "tag": None},
    }


@pytest.fixture(scope="module")
def temp_integration_dir():
    """Create a temporary directory for integration tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator with proper responses for testing"""
    mock_orch = Mock()
    mock_orch.get_session_counter.return_value = 1
    mock_orch._provider.provider_type = "openai"
    mock_orch._provider.get_default_llm_identity.return_value = LLMIdentity(
        "gpt-5-nano", provider_type="openai"
    )
    mock_orch._backend.retrieve.return_value = None  # No cache by default

    # Set up the metadata dictionary structure that my_metadata property expects
    mock_orch._fm.metadata = {"agents": {}}

    mock_orch._logger.info = Mock()

    # Create a mock call ID for responses
    mock_call_id = {
        "agent_name": "test_agent",
        "doc_hash": "test_hash",
        "seq_id": 0,
        "session_id": 1,
        "meta": {"provider_type": "openai", "tag": None},
    }

    mock_orch._provider.submit_query_to_provider.return_value = ReadyLLMResponse(
        call_id=mock_call_id, value="Mock response"
    )

    mock_orch._backend.submit_query.return_value = ReadyLLMResponse(
        call_id=mock_call_id, value="Mock response"
    )

    yield mock_orch
