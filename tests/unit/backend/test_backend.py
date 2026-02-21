"""
Unit tests for backend functionality

Tests the backend implementations including:
- AsyncBackend initialization, task management, and lifecycle
- SyncBackend initialization and synchronous operations
- Backend persistence and data storage
- Backend retrieve operations
- Backend shutdown and cleanup
- AsyncBackend functionality after persist() calls
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from parallellm.core.backend.concurrent_backend import ConcurrentBackend
from parallellm.core.backend.sync_backend import SyncBackend
from parallellm.file_io.file_manager import FileManager
from parallellm.types import ParsedResponse, to_serial_id


pytest.skip("Takes ~1s", allow_module_level=True)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def file_manager(temp_dir):
    """Create a FileManager instance for testing"""
    return FileManager(temp_dir)


@pytest.fixture
def sample_call_id():
    """Create a sample CallIdentifier for testing"""
    return {
        "agent_name": "test_agent",
        "doc_hash": "test_hash_123",
        "seq_id": 1,
        "session_id": 1,
        "meta": {"provider_type": "openai", "tag": None},
    }


class TestConcurrentBackend:
    """Test ConcurrentBackend functionality"""

    def test_concurrent_backend_submit_coro(self, file_manager, sample_call_id):
        """Test submitting coroutines to ConcurrentBackend"""
        backend = ConcurrentBackend(file_manager)

        async def sample_coro():
            await asyncio.sleep(0.01)
            return {"role": "assistant", "content": "test_result"}

        # Submit the coroutine
        future = backend.submit_coro(sample_call_id, sample_coro())

        assert future is not None
        assert len(backend.tasks) == 1
        assert len(backend.task_metas) == 1
        assert backend.task_metas[0] == sample_call_id

        # retrieve() will wait for completion and verify result
        result = backend.retrieve(sample_call_id)
        assert result is not None
        assert "test_result" in result.text

        # Clean up
        backend.shutdown()

    def test_concurrent_backend_shutdown(self, file_manager):
        """Test ConcurrentBackend shutdown functionality"""
        backend = ConcurrentBackend(file_manager)

        assert backend._loop is not None
        assert backend._loop_thread.is_alive()

        # Shutdown the backend
        backend.shutdown()

        # Verify shutdown
        assert backend._shutdown_event.is_set()
        # Give some time for thread to clean up
        backend._loop_thread.join(timeout=5.0)
        assert not backend._loop_thread.is_alive()

    def test_concurrent_backend_persist(self, file_manager, sample_call_id):
        """Test ConcurrentBackend persist functionality"""
        backend = ConcurrentBackend(file_manager)

        async def sample_coro():
            return {"role": "assistant", "content": "persist test"}

        # Submit a task
        backend.submit_coro(sample_call_id, sample_coro())

        # Call persist - should wait for tasks to complete
        backend.persist(timeout=5.0)

        # Backend should still be functional for new operations
        assert backend._loop is not None
        assert not backend._loop.is_closed()

        # Clean up
        backend.shutdown()

    def test_concurrent_backend_functionality_after_persist(
        self, file_manager, sample_call_id
    ):
        """Test that ConcurrentBackend remains functional after calling persist()"""
        backend = ConcurrentBackend(file_manager)

        # Submit initial task
        async def first_coro():
            await asyncio.sleep(0.01)
            return {"role": "assistant", "content": "first response"}

        backend.submit_coro(sample_call_id, first_coro())

        # Call persist to wait for tasks to complete
        backend.persist(timeout=5.0)

        # Verify we can retrieve the first result
        first_result = backend.retrieve(sample_call_id)
        assert first_result is not None
        assert "first response" in first_result.text

        # Verify backend is still functional by submitting another task
        second_call_id = sample_call_id.copy()
        second_call_id["seq_id"] = 2
        second_call_id["doc_hash"] = "test_hash_456"

        async def second_coro():
            await asyncio.sleep(0.01)
            return {"role": "assistant", "content": "second response"}

        # This should not raise an exception
        future = backend.submit_coro(second_call_id, second_coro())
        assert future is not None

        # Backend should still have the event loop running
        assert backend._loop is not None
        assert not backend._loop.is_closed()
        assert backend._loop_thread.is_alive()

        # Retrieve second result
        second_result = backend.retrieve(second_call_id)
        assert second_result is not None
        assert "second response" in second_result.text

        # Clean up
        backend.shutdown()

    def test_concurrent_backend_does_not_wait_for_unrelated_tasks(
        self, file_manager, sample_call_id
    ):
        """Test that retrieve() for one task doesn't wait for unrelated slow tasks"""
        backend = ConcurrentBackend(file_manager)

        # Submit a very slow task first
        slow_call_id = sample_call_id.copy()
        slow_call_id["doc_hash"] = "very_slow_hash"

        async def very_slow_coro():
            await asyncio.sleep(2.0)  # Very slow
            return {"role": "assistant", "content": "very slow response"}

        # Submit a fast task
        fast_call_id = sample_call_id.copy()
        fast_call_id["doc_hash"] = "very_fast_hash"
        fast_call_id["seq_id"] = 2

        async def very_fast_coro():
            return {"role": "assistant", "content": "very fast response"}  # Immediate

        backend.submit_coro(slow_call_id, very_slow_coro())
        backend.submit_coro(fast_call_id, very_fast_coro())

        # Retrieve the fast task - should complete quickly without waiting for slow task
        import time

        start_time = time.time()
        fast_result = backend.retrieve(fast_call_id)
        elapsed_time = time.time() - start_time

        assert fast_result is not None
        assert "very fast response" in fast_result.text
        assert elapsed_time < 0.1  # Should be very fast, not waiting 2 seconds

        # The slow task should still be running
        assert len(backend.tasks) == 1

        # Clean up - shutdown will clean up the slow task
        backend.shutdown()


class TestSyncBackend:
    """Test SyncBackend functionality"""

    def test_sync_backend_submit_sync_call(self, file_manager, sample_call_id):
        """Test submitting synchronous calls to SyncBackend"""
        backend = SyncBackend(file_manager)

        def sample_sync_function():
            return {"role": "assistant", "content": "sync response"}

        # Submit the call
        result = backend.submit_sync_call(sample_call_id, sample_sync_function)

        assert result is not None
        assert len(result) == 3  # resp_text, resp_id, resp_metadata

        resp_text, resp_id, resp_metadata = result
        assert isinstance(resp_text, str)
        assert "sync response" in resp_text
        assert resp_metadata is not None

        # Check that result is stored in pending results
        key = to_serial_id(sample_call_id, add_sess=False)
        assert key in backend._pending_results
        assert backend._pending_results[key] == resp_text

    def test_sync_backend_retrieve(self, file_manager, sample_call_id):
        """Test retrieving data from SyncBackend"""
        backend = SyncBackend(file_manager)

        def sample_sync_function():
            return {"role": "assistant", "content": "retrieve test"}

        # Submit and store a call
        backend.submit_sync_call(sample_call_id, sample_sync_function)

        # Retrieve the result
        result = backend.retrieve(sample_call_id)

        assert result is not None
        assert isinstance(result, ParsedResponse)
        assert "retrieve test" in result.text

    def test_sync_backend_persist(self, file_manager, sample_call_id):
        """Test SyncBackend persist functionality"""
        backend = SyncBackend(file_manager)

        def sample_sync_function():
            return {"role": "assistant", "content": "persist test"}

        # Submit a call
        backend.submit_sync_call(sample_call_id, sample_sync_function)

        # Call persist - should complete without error
        backend.persist()

        # Backend should still be functional
        assert backend._ds is not None
