from pathlib import Path
import tempfile
from threading import Lock
from unittest.mock import Mock
import logging
import pytest
from parallem.core.gateway import resume_directory
from parallem.core.response import ReadyLLMResponse
from parallem.core.file_manager import FileManager
from parallem.core.agent.orchestrator import AgentOrchestrator
from parallem.core.backend.sync_backend import SyncBackend
from parallem.logging.dash_logger import PrimitiveDashboardLogger
from parallem.testing.simple_backend import MockBackend
from parallem.testing.simple_mock import mock_openai_client
from parallem.types import CallIdentifier, LLMIdentity


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
    seq_counters = {}
    mock_orch.get_session_counter.return_value = 1
    mock_orch._dashlog = PrimitiveDashboardLogger()
    mock_orch._provider.provider_type = "openai"
    mock_orch._provider.get_default_llm_identity.return_value = LLMIdentity(
        "gpt-5-nano", provider_type="openai"
    )
    mock_orch._backend.retrieve.return_value = None  # No cache by default

    def _next_seq_id(agent_name: str) -> int:
        key = (mock_orch.get_session_counter.return_value, agent_name)
        current = seq_counters.get(key, 0)
        seq_counters[key] = current + 1
        return current

    mock_orch.next_seq_id = Mock(side_effect=_next_seq_id)

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


@pytest.fixture(scope="session")
def lock():
    return Lock()


@pytest.fixture(scope="session")
def session_orch_root():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="session")
def async_sync_orch(session_orch_root):
    fm = FileManager(session_orch_root / "async-sync")
    orch = AgentOrchestrator(
        file_manager=fm,
        backend=MockBackend(),
        provider=object(),
        logger=logging.getLogger("parallem.test"),
        dashlog=PrimitiveDashboardLogger(),
        strategy="sync",
    )
    yield orch
    orch.finalize_and_persist()


@pytest.fixture(scope="session")
def async_concurrent_orch(session_orch_root):
    fm = FileManager(session_orch_root / "async-concurrent")
    orch = AgentOrchestrator(
        file_manager=fm,
        backend=MockBackend(),
        provider=object(),
        logger=logging.getLogger("parallem.test"),
        dashlog=PrimitiveDashboardLogger(),
        strategy="concurrent",
    )
    yield orch
    orch.finalize_and_persist()


@pytest.fixture(scope="session")
def async_batch_orch(session_orch_root):
    fm = FileManager(session_orch_root / "async-batch")
    orch = AgentOrchestrator(
        file_manager=fm,
        backend=MockBackend(),
        provider=object(),
        logger=logging.getLogger("parallem.test"),
        dashlog=PrimitiveDashboardLogger(),
        strategy="batch",
    )
    yield orch
    orch.finalize_and_persist()


@pytest.fixture(scope="session")
def persistence_sync_orch(session_orch_root):
    fm = FileManager(session_orch_root / "persistence-sync")
    backend = SyncBackend(fm)
    orch = AgentOrchestrator(
        file_manager=fm,
        backend=backend,
        provider=Mock(),
        logger=Mock(),
        dashlog=PrimitiveDashboardLogger(),
    )
    yield orch
    orch.finalize_and_persist()


@pytest.fixture(scope="session")
def persistence_mock_backend_orch(session_orch_root):
    fm = FileManager(session_orch_root / "persistence-mock")
    backend = MockBackend()
    orch = AgentOrchestrator(
        file_manager=fm,
        backend=backend,
        provider=Mock(),
        logger=Mock(),
        dashlog=PrimitiveDashboardLogger(),
    )
    yield orch
    orch.finalize_and_persist()


@pytest.fixture(scope="session")
def persistence_ignore_cache_orch(session_orch_root):
    fm = FileManager(session_orch_root / "persistence-ignore-cache")
    backend = Mock()
    backend.retrieve.return_value = "cached_response"
    orch = AgentOrchestrator(
        file_manager=fm,
        backend=backend,
        provider=Mock(),
        logger=Mock(),
        dashlog=PrimitiveDashboardLogger(),
        ignore_cache=True,
    )
    yield orch
    orch.finalize_and_persist()


@pytest.fixture(scope="session")
def shared_sync_orch(session_orch_root):
    orch_dir = session_orch_root / "shared-sync"
    mock_client = mock_openai_client()
    orch = resume_directory(
        orch_dir,
        provider="openai",
        strategy="sync",
        client=mock_client,
        hash_by=["llm"],
    )
    orch._mock_client = mock_client
    yield orch
    orch.finalize_and_persist()


@pytest.fixture(scope="session")
def shared_concurrent_orch(session_orch_root):
    orch_dir = session_orch_root / "shared-concurrent"
    mock_client = mock_openai_client(concurrent=True)
    orch = resume_directory(
        orch_dir,
        provider="openai",
        strategy="concurrent",
        client=mock_client,
    )
    orch._mock_client = mock_client
    yield orch
    orch.finalize_and_persist()
