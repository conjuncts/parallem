from pathlib import Path
from unittest.mock import Mock

import pytest

from parallem.core.agent.orchestrator import AgentOrchestrator
from parallem.core.exception import NotAvailable
from parallem.core.file_manager import FileManager
from parallem.logging.dash_logger import PrimitiveDashboardLogger
from parallem.testing.simple_backend import MockBackend


@pytest.fixture
def orchestrator(session_orch_root):
    """Create a lightweight orchestrator for context manager tests."""
    fm = FileManager(Path(session_orch_root) / "orchestrator-context")
    return AgentOrchestrator(
        file_manager=fm,
        backend=MockBackend(),
        provider=Mock(),
        logger=Mock(),
        dashlog=PrimitiveDashboardLogger(),
        strategy="batch",
    )


def test_unexpected_exception_skips_finalize(orchestrator):
    """Unexpected exceptions should surface without finalizing batch work."""
    orchestrator.finalize_tasks = Mock()

    with pytest.raises(RuntimeError):
        with orchestrator:
            raise RuntimeError("boom")

    orchestrator.finalize_tasks.assert_not_called()


def test_parallem_signal_still_finalizes(orchestrator):
    """Expected parallem signals should still trigger finalization."""
    orchestrator.finalize_tasks = Mock()

    with pytest.raises(NotAvailable):
        with orchestrator:
            raise NotAvailable("pending")

    orchestrator.finalize_tasks.assert_called_once()
