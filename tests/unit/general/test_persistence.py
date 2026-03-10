"""
Unit tests for userdata persistence and FileManager

Tests the userdata persistence functionality including:
- FileManager initialization and metadata handling
- save_userdata and load_userdata operations
- Agent metadata management
- File sanitization and directory allocation
- Session counter and lock file handling
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock
from pipelinellm.core.file_manager import FileManager
from pipelinellm.core.agent.orchestrator import AgentOrchestrator
from pipelinellm.core.response import ReadyLLMResponse, PendingLLMResponse
from pipelinellm.testing.simple_backend import MockBackend
from pipelinellm.types import (
    ParsedResponse,
)


class TestFileManagerBasics:
    """Test basic FileManager functionality"""

    def test_file_manager_creation(self):
        """Test creating a FileManager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            assert fm.directory == Path(temp_dir)
            assert fm.metadata_file == Path(temp_dir) / "metadata.json"
            assert fm.lock_file == Path(temp_dir) / ".filemanager.lock"

            # Directory should be created
            assert Path(temp_dir).exists()

            # Lock file should be created
            assert fm.lock_file.exists()

    def test_metadata_initialization(self):
        """Test metadata is properly initialized"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            # Should have default metadata structure
            assert "agents" in fm.metadata
            assert "session_counter" in fm.metadata
            assert "" in fm.metadata["agents"]

            # Default agent should have proper structure
            default_agent = fm.metadata["agents"][""]
            assert default_agent == {}

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

    def test_metadata_persistence_across_instances(self):
        """Test that metadata persists across FileManager instances"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First instance - modify metadata
            fm1 = FileManager(temp_dir)
            fm1.metadata["agents"]["test_agent"] = {
                "foo": "bar",
                "foo_counter": 5,
            }
            fm1.persist()

            # Second instance should load persisted metadata
            fm2 = FileManager(temp_dir)
            assert "test_agent" in fm2.metadata["agents"]
            assert fm2.metadata["agents"]["test_agent"]["foo"] == "bar"
            assert fm2.metadata["agents"]["test_agent"]["foo_counter"] == 5

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


class TestUserdataPersistence:
    """Test userdata save/load functionality"""

    def test_save_and_load_simple_data(self):
        """Test saving and loading simple data types"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            # Test different data types
            test_data = {
                "string": "Hello World",
                "number": 42,
                "list": [1, 2, 3],
                "dict": {"key": "value"},
            }

            for key, value in test_data.items():
                fm.save_userdata(key, value)
                loaded_value = fm.load_userdata(key)
                assert loaded_value == value

    def test_save_userdata_creates_directory(self):
        """Test that save_userdata creates userdata directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            userdata_dir = fm.directory / "userdata"
            assert not userdata_dir.exists()

            fm.save_userdata("test_key", "test_value")

            assert userdata_dir.exists()
            assert userdata_dir.is_dir()

    def test_load_nonexistent_data(self):
        """Test loading nonexistent data raises FileNotFoundError"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            with pytest.raises(FileNotFoundError):
                fm.load_userdata("nonexistent_key")

    def test_save_userdata_overwrite_protection(self):
        """Test overwrite protection in save_userdata"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            # Save initial data
            fm.save_userdata("test_key", "original_value")

            # No overwrite (should not change)
            fm.save_userdata("test_key", "new_value", overwrite=False)
            loaded_value = fm.load_userdata("test_key")
            assert loaded_value == "original_value"

            # Overwrite (should change)
            fm.save_userdata("test_key", "new_value", overwrite=True)
            loaded_value = fm.load_userdata("test_key")
            assert loaded_value == "new_value"

    def test_userdata_file_naming(self):
        """Test that userdata files are named correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            fm.save_userdata("test/key", "test_value")

            # Check that file was created with sanitized name
            userdata_dir = fm.directory / "userdata"
            pkl_files = list(userdata_dir.glob("*.pkl"))

            assert len(pkl_files) == 1
            assert "test_key" in pkl_files[0].name  # Should be sanitized


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

    def test_orchestrator_userdata_operations(self):
        """Test userdata operations through AgentOrchestrator"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            orchestrator = AgentOrchestrator(
                file_manager=fm,
                backend=Mock(),
                provider=Mock(),
                logger=Mock(),
                dashlog=Mock(),
            )

            # Test save/load through orchestrator
            test_data = {"key": "value", "number": 42}
            orchestrator.userdata["test_data"] = test_data

            loaded_data = orchestrator.userdata["test_data"]
            assert loaded_data == test_data

    def test_userdata_llm_responses(self, generic_call_id):
        """Test that orchestrator injects backend into LLMResponses"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            backend = MockBackend()

            orchestrator = AgentOrchestrator(
                file_manager=fm,
                backend=backend,
                provider=Mock(),
                logger=Mock(),
                dashlog=Mock(),
            )

            pr = ParsedResponse(
                text="backend_test_value",
                response_id="resp_123",
                metadata=None,
            )
            backend.store(generic_call_id, pr)

            pending_response = PendingLLMResponse(
                call_id=generic_call_id, backend=backend
            )
            ready_response = ReadyLLMResponse(
                call_id=generic_call_id, value="test_value"
            )

            # Save responses
            orchestrator.userdata["pending"] = pending_response
            orchestrator.userdata["ready"] = ready_response

            # Load and verify backend injection
            loaded_pending = orchestrator.userdata["pending"]
            loaded_ready = orchestrator.userdata["ready"]

            assert isinstance(loaded_pending, PendingLLMResponse)
            assert loaded_pending._backend == backend
            assert loaded_pending.resolve() == "backend_test_value"

            assert isinstance(loaded_ready, ReadyLLMResponse)
            # Both depend solely on backend
            assert loaded_ready.resolve() == "backend_test_value"

    def test_orchestrator_ignore_cache_parameter(self, generic_call_id):
        """Test that ignore_cache parameter works correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)
            mock_backend = Mock()
            mock_backend.retrieve.return_value = "cached_response"

            mock_backend.submit_query.return_value = ReadyLLMResponse(
                call_id=generic_call_id, value="fresh_response"
            )

            # Create orchestrator with ignore_cache=True
            orchestrator = AgentOrchestrator(
                file_manager=fm,
                backend=mock_backend,
                provider=Mock(),
                logger=Mock(),
                dashlog=Mock(),
                ignore_cache=True,
            )

            with orchestrator.agent() as agent:
                response = agent.ask_llm("Test prompt")

                assert response.resolve() == "fresh_response"


class TestFileManagerPersistence:
    """Test FileManager persist functionality"""

    def test_persist_saves_metadata(self):
        """Test that persist saves metadata to disk"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            # Modify metadata
            fm.metadata["agents"]["new_agent"] = {
                "foo": "bar",
                "foo_counter": 10,
            }

            fm.persist()

            # Verify metadata file contains changes
            with open(fm.metadata_file, "r") as f:
                saved_metadata = json.load(f)

            assert "new_agent" in saved_metadata["agents"]
            assert saved_metadata["agents"]["new_agent"]["foo"] == "bar"
            assert saved_metadata["agents"]["new_agent"]["foo_counter"] == 10

    def test_persist_idempotent(self):
        """Test that multiple persist calls are safe"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            # Multiple persist calls should not cause issues
            fm.persist()
            fm.persist()
            fm.persist()

            # Metadata should still be valid
            assert "agents" in fm.metadata
            assert "session_counter" in fm.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
