"""
Unit tests for userdata persistence and FileManager

Tests the userdata persistence functionality including:
- FileManager initialization and metadata handling
- Agent metadata management
- File sanitization and directory allocation
- Session counter and lock file handling
"""

import pytest
import tempfile
import multiprocessing
import queue
from pathlib import Path
from uuid import uuid4
from parallem.core.file_manager import FileManager
from parallem.core.response import ReadyLLMResponse, PendingLLMResponse
from parallem.types import (
    ParsedResponse,
)


@pytest.fixture
def test_agent_name(request):
    """Generate a unique agent name per test for isolation."""
    return f"{request.node.name}-{uuid4().hex}"


@pytest.fixture
def test_userdata_key(request):
    """Generate a unique userdata key per test for isolation."""
    return f"userdata-{request.node.name}-{uuid4().hex}"


def _init_file_manager_and_get_session_counter(temp_dir: str, result_queue):
    fm = FileManager(temp_dir)
    result_queue.put(fm.metadata["session_counter"])
    fm.persist()
    fm._cleanup()


class TestFileManagerBasics:
    """Test basic FileManager functionality"""

    def test_file_manager_creation(self):
        """Test creating a FileManager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            assert fm.directory == Path(temp_dir)
            assert fm.lock_file == Path(temp_dir) / ".filemanager.lock"

            # Directory should be created
            assert Path(temp_dir).exists()

            # Lock file should be created
            assert fm.lock_file.exists()

    def test_metadata_initialization(self):
        """Test metadata is properly initialized"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            # Should expose session counter in metadata
            assert "session_counter" in fm.metadata

    def test_session_counter_increments(self):
        """Test session counter increments on each new FileManager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First instance
            fm1 = FileManager(temp_dir)
            session1 = fm1.metadata["session_counter"]
            fm1.persist()

            # Second instance should increment
            fm2 = FileManager(temp_dir)
            session2 = fm2.metadata["session_counter"]

            assert session2 == session1 + 1

    @pytest.mark.skip("slow")
    def test_session_counter_multiprocess_unique(self):
        """Test session counter remains unique across concurrent processes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            process_count = 8
            ctx = multiprocessing.get_context("spawn")
            result_queue = ctx.Queue()

            processes = [
                ctx.Process(
                    target=_init_file_manager_and_get_session_counter,
                    args=(temp_dir, result_queue),
                )
                for _ in range(process_count)
            ]

            for process in processes:
                process.start()

            for process in processes:
                process.join(timeout=15)
                assert process.exitcode == 0

            session_ids = []
            for _ in range(process_count):
                try:
                    session_ids.append(result_queue.get(timeout=2))
                except queue.Empty:
                    pytest.fail("Timed out receiving session IDs from worker processes")

            assert len(set(session_ids)) == process_count

    def test_lock_file_cleanup(self):
        """Test lock file is cleaned up"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)
            lock_file = fm.lock_file

            assert lock_file.exists()

            # Cleanup should remove lock file
            fm._cleanup()
            assert not lock_file.exists()


class TestSanitization:
    """Test input sanitization functionality"""

    def test_sanitize_basic(self):
        """Test basic string sanitization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            result = fm._sanitize("simple_name")
            assert "simple_name" in result
            assert len(result.split("-")) == 2  # name-hash format

    def test_sanitize_none_input(self):
        """Test sanitization with None input"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            result = fm._sanitize(None)
            assert result == "default"

            result_custom = fm._sanitize(None, default="custom")
            assert result_custom == "custom"

    def test_sanitize_special_characters(self):
        """Test sanitization removes special characters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            result = fm._sanitize("test/name!@#$%")
            # Should replace special chars with underscores
            assert "/" not in result
            assert "!" not in result
            assert "@" not in result

    def test_sanitize_no_hash(self):
        """Test sanitization without hash"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            result = fm._sanitize("test_name", add_hash=False)
            assert result == "test_name"
            assert "-" not in result

    def test_sanitize_long_string(self):
        """Test sanitization with long strings"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            long_name = "a" * 100  # 100 characters
            result = fm._sanitize(long_name)

            # Should be truncated to 64 chars + hash
            name_part = result.split("-")[0]
            assert len(name_part) <= 64

    def test_sanitize_empty_string(self):
        """Test sanitization with empty string"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            result = fm._sanitize("")
            assert "empty" in result  # Default fallback


class TestDatastoreAllocation:
    """Test datastore directory allocation"""

    def test_allocate_datastore_basic(self):
        """Test basic datastore directory retrieval"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            path = fm.path_datastore()

            # Should create base datastore directory
            expected_path = Path(temp_dir) / "datastore"

            assert path.exists()
            assert path.is_dir()
            assert path == expected_path

    def test_get_datastore_directory_creates_parents(self):
        """Test that get_datastore_directory creates the directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            datastore_base = Path(temp_dir) / "datastore"
            assert not datastore_base.exists()

            path = fm.path_datastore()

            assert datastore_base.exists()
            assert path == datastore_base


class TestAgentOrchestratorIntegration:
    """Test integration with AgentOrchestrator"""

    def test_msg_state_persist_snapshot_replay_overwrites_local_changes(
        self, persistence_sync_orch, test_agent_name
    ):
        """Test non-continued persist stores a snapshot log that can be replayed safely."""
        with persistence_sync_orch.agent(test_agent_name) as agent:
            msg_state = agent.get_msg_state()
            msg_state.append("saved")
            msg_state.save()
            msg_state.append("unsaved")

            loaded = msg_state.load()
            assert list(loaded) == ["saved"]

    def test_msg_state_load_requires_manual_persist(self, persistence_sync_orch, test_agent_name):
        """Test load mode state persists only when MessageState.persist() is called."""
        with persistence_sync_orch.agent(test_agent_name) as agent:
            msg_state = agent.get_msg_state().load()
            msg_state.append("hello")
            msg_state.save()

        with persistence_sync_orch.agent(test_agent_name) as agent:
            loaded = agent.get_msg_state().load()
            assert list(loaded) == ["hello"]

    def test_msg_state_continued_replay_accumulates(self, persistence_sync_orch, test_agent_name):
        """Test continued mode replays recorded operations each separate run."""
        lengths = []

        with persistence_sync_orch.agent(test_agent_name) as agent:
            msg_state = agent.get_msg_state().load()
            msg_state.append("x")
            lengths.append(len(msg_state))
            msg_state.save()

        with persistence_sync_orch.agent(test_agent_name) as agent:
            msg_state = agent.get_msg_state().load()
            msg_state.append("x")
            lengths.append(len(msg_state))
            msg_state.save()

        with persistence_sync_orch.agent(test_agent_name) as agent:
            msg_state = agent.get_msg_state().load()
            msg_state.append("x")
            lengths.append(len(msg_state))
            msg_state.save()

        assert lengths == [1, 2, 3]

    def test_orchestrator_userdata_operations(
        self, persistence_mock_backend_orch, test_userdata_key
    ):
        """Test userdata operations through AgentOrchestrator"""
        test_data = {"key": "value", "number": 42}
        persistence_mock_backend_orch.userdata[test_userdata_key] = test_data

        loaded_data = persistence_mock_backend_orch.userdata[test_userdata_key]
        assert loaded_data == test_data

    def test_userdata_llm_responses(
        self,
        generic_call_id,
        persistence_mock_backend_orch,
        test_userdata_key,
    ):
        """Test that orchestrator injects backend into LLMResponses"""
        backend = persistence_mock_backend_orch._backend

        pr = ParsedResponse(
            text="backend_test_value",
            response_id="resp_123",
            metadata=None,
        )
        backend.store(generic_call_id, pr)

        pending_response = PendingLLMResponse(call_id=generic_call_id, backend=backend)
        ready_response = ReadyLLMResponse(call_id=generic_call_id, value="test_value")

        pending_key = f"{test_userdata_key}-pending"
        ready_key = f"{test_userdata_key}-ready"
        persistence_mock_backend_orch.userdata[pending_key] = pending_response
        persistence_mock_backend_orch.userdata[ready_key] = ready_response

        loaded_pending = persistence_mock_backend_orch.userdata[pending_key]
        loaded_ready = persistence_mock_backend_orch.userdata[ready_key]

        assert isinstance(loaded_pending, PendingLLMResponse)
        assert loaded_pending._backend == backend
        assert loaded_pending.final_answer == "backend_test_value"

        assert isinstance(loaded_ready, ReadyLLMResponse)
        assert loaded_ready.final_answer == "test_value"

    def test_orchestrator_ignore_cache_parameter(
        self, generic_call_id, persistence_ignore_cache_orch, test_agent_name
    ):
        """Test that ignore_cache parameter works correctly"""
        mock_backend = persistence_ignore_cache_orch._backend
        mock_backend.submit_query.return_value = ReadyLLMResponse(
            call_id=generic_call_id, value="fresh_response"
        )

        with persistence_ignore_cache_orch.agent(test_agent_name) as agent:
            response = agent.ask_llm("Test prompt")

            assert response.final_answer == "fresh_response"


class TestFileManagerPersistence:
    """Test FileManager persist functionality"""

    def test_persist_idempotent(self):
        """Test that multiple persist calls are safe"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            # Multiple persist calls should not cause issues
            fm.persist()
            fm.persist()
            fm.persist()

            # Metadata should still be valid
            assert "session_counter" in fm.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
