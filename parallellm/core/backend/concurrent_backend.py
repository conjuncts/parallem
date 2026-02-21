import asyncio
import types
import threading
import atexit
from typing import Optional, TYPE_CHECKING
from parallellm.core.backend import BaseBackend
from parallellm.core.throttler import Throttler
from parallellm.core.calls import _call_matches
from parallellm.core.datastore.sqlite import SQLiteDatastore
from parallellm.core.response import PendingLLMResponse
from parallellm.file_io.file_manager import FileManager
from parallellm.logging.dash_logger import (
    DashboardLogger,
    HashStatus,
    PrimitiveDashboardLogger,
)
from parallellm.types import (
    CallIdentifier,
    CommonQueryParameters,
    ParsedResponse,
    CommonQueryParameters,
)

if TYPE_CHECKING:
    from parallellm.provider.base import ConcurrentProvider


class ConcurrentBackend(BaseBackend):
    """
    A backend is a data store, but also a way to poll.
    This backend owns its own event loop running in a separate thread.

    The Datastore
    """

    def __init__(
        self,
        fm: FileManager,
        dashlog: DashboardLogger = PrimitiveDashboardLogger(),
        *,
        datastore_cls=None,
        rewrite_cache: bool = False,
        max_concurrent: int = 20,
        throttler=None,
    ):
        """
        Initialize the ConcurrentBackend.

        :param fm: FileManager for data persistence
        :param dashlog: Optional dashboard logger for monitoring
        :param datastore_cls: Custom datastore class (defaults to SQLiteDatastore)
        :param rewrite_cache: Whether to overwrite existing cache entries
        :param max_concurrent: Maximum number of concurrent tasks
        :param throttler: Throttler instance for rate limiting (default: None)
        """
        self._fm = fm
        self.dashlog = dashlog
        self._rewrite_cache = rewrite_cache
        self._max_concurrent = max_concurrent

        if throttler is not None:
            self._throttler = throttler
        else:
            # No rate limiting
            self._throttler = Throttler(
                max_requests_per_window=None,
                window_seconds=None,
            )

        # These should ONLY be accessed from the event loop thread
        self.tasks: list[asyncio.Task] = []
        self.task_metas: list[dict] = []
        self._loop: asyncio.BaseEventLoop = None
        self._loop_thread = None
        self._shutdown_event = threading.Event()
        self._loop_ready_event = threading.Event()

        # Start the event loop in a separate thread
        self.datastore_cls = datastore_cls
        self._concurrent_ds: Optional[SQLiteDatastore] = None
        self._start_event_loop()

        # Register cleanup to run on program exit
        atexit.register(self.shutdown)

    def _get_datastore(self):
        return self._concurrent_ds

    def _start_event_loop(self):
        """Start the event loop in a separate thread"""

        def run_event_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # Initialize the datastore now that the loop is running
            if self.datastore_cls is None:
                self._concurrent_ds = SQLiteDatastore(self._fm)
            else:
                self._concurrent_ds = self.datastore_cls(self._fm)

            # Signal that the loop is ready
            self._loop_ready_event.set()

            try:
                # Keep the loop running until shutdown
                self._loop.run_until_complete(self._wait_for_shutdown())
            finally:
                # Clean up any remaining tasks
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                self._loop.close()

        self._loop_thread = threading.Thread(target=run_event_loop, daemon=True)
        self._loop_thread.start()

        # Wait for the loop to be ready (blocks until event is set)
        self._loop_ready_event.wait()

    async def _wait_for_shutdown(self):
        """Keep the event loop running until shutdown is requested"""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(0.1)

        # Clean up the datastore from the same thread that created it
        if hasattr(self, "_concurrent_ds"):
            self._concurrent_ds.close()
            del self._concurrent_ds

    def _run_coroutine(self, coro):
        """Run a coroutine in the backend's event loop and return the result"""
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("ConcurrentBackend event loop is not running")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        res = future.result()
        return res

    async def _apply_throttling(self) -> None:
        """Apply throttling by waiting if necessary"""
        delay = self._throttler.calculate_delay()
        if delay > 0:
            await asyncio.sleep(delay)
            # After waiting, record the actual submission time
            if self._throttler.is_enabled():
                self._throttler.record_request()

    def submit_query(
        self,
        provider: "ConcurrentProvider",
        params: CommonQueryParameters,
        *,
        call_id: CallIdentifier,
        **kwargs,
    ) -> PendingLLMResponse:
        """
        New control flow: Backend calls provider to get coroutine, then executes it.
        This inverts control from provider calling backend.
        """

        coro = provider.prepare_concurrent_call(
            params,
            **kwargs,
        )

        # Submit for concurrent execution
        self.submit_coro(call_id=call_id, coro=coro, provider=provider)

        return PendingLLMResponse(
            call_id=call_id,
            backend=self,
        )

    def shutdown(self):
        """Shutdown the event loop and cleanup"""
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)

    async def _cleanup_datastore(self):
        """Clean up the datastore from the concurrent thread"""
        if hasattr(self, "_concurrent_ds"):
            self._concurrent_ds.close()
            # del self._concurrent_ds

    def cleanup_datastore_sync(self):
        """Synchronously trigger datastore cleanup in the concurrent thread"""
        if self._loop is not None and not self._loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(
                self._cleanup_datastore(), self._loop
            )
            try:
                future.result(timeout=5.0)
            except Exception as e:
                print(f"Warning: Failed to cleanup datastore: {e}")

    def submit_coro(
        self, call_id: CallIdentifier, coro: types.CoroutineType, provider=None
    ):
        """Submit a coroutine to be executed in the backend's event loop"""
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("ConcurrentBackend event loop is not running")

        # Need to poll.
        # NOTE: modifying self.tasks in general is NOT thread-safe,
        # But here we are only checking its length.
        # If we push the state mutator onto the event loop then it's OK
        if len(self.tasks) >= self._max_concurrent:
            # Wait for the oldest task to complete
            oldest_meta = self.task_metas[0]
            self._run_coroutine(self._poll_changes(oldest_meta))

        # Create the task in the backend's event loop
        future = asyncio.run_coroutine_threadsafe(
            self._create_and_store_task(call_id, coro, provider), self._loop
        )

        self.dashlog.update_hash(call_id["doc_hash"], HashStatus.SENT)
        # Don't wait for the result, just submit it
        return future

    async def _create_and_store_task(
        self, call_id: CallIdentifier, coro: types.CoroutineType, provider=None
    ):
        """Helper to create and store a task in the event loop"""

        # Wrap the coro to parse response immediately
        metadata = call_id.copy()

        async def wrapped_coro():
            # Apply throttling before creating the task
            await self._apply_throttling()

            result = await coro
            parsed = provider.parse_response(result)
            return parsed, metadata

        task = asyncio.create_task(wrapped_coro())
        self.tasks.append(task)
        self.task_metas.append(metadata)
        return task

    async def _poll_changes(self, until_call_id: Optional[CallIdentifier]):
        """
        A chance to poll for changes and update the data store
        """
        # collect as results come in
        # need to keep track of which ones we process (otherwise, race condition)
        done_tasks = []
        for coro in asyncio.as_completed(self.tasks):
            parsed, metadata = await coro

            call_id: CallIdentifier = metadata.copy()

            self._concurrent_ds.store(call_id, parsed, upsert=self._rewrite_cache)
            done_tasks.append(metadata)

            self.dashlog.update_hash(call_id["doc_hash"], HashStatus.RECEIVED)

            # Stop if we reached the target
            if until_call_id is not None and _call_matches(until_call_id, call_id):
                break

        # pop completed tasks
        for i in reversed(range(len(self.tasks))):
            meta = self.task_metas[i]
            if meta in done_tasks:
                self.tasks.pop(i)
                self.task_metas.pop(i)
                # print(f"Completed {meta['doc_hash'][:8]}:{meta['seq_id']}")

    async def aretrieve(
        self, call_id: CallIdentifier, metadata=False
    ) -> Optional[ParsedResponse]:
        # only poll for changes if we have a matching task
        if any(_call_matches(m, call_id) for m in self.task_metas):
            await self._poll_changes(call_id)
        return self._concurrent_ds.retrieve(call_id, metadata=metadata)

    def persist(self, timeout=30.0):
        """
        Synchronous persist that uses the backend's event loop.
        Cleans up any datastore resources.
        """

        # We want to wait for all pending tasks to complete
        if self._loop is not None and not self._loop.is_closed():
            # Create a dummy TaskIdentifier for _poll_changes - we'll pass None values to poll all
            future = asyncio.run_coroutine_threadsafe(
                self._poll_changes(None), self._loop
            )
            try:
                future.result(timeout=timeout)
            except Exception as e:
                print(f"Warning: Failed to wait for pending tasks: {e}")

        # Let datastore cleanup
        self._concurrent_ds.persist()

        # Close datastore connections to ensure proper cleanup, especially important on Windows
        self.cleanup_datastore_sync()

    def retrieve(
        self, call_id: CallIdentifier, metadata=False
    ) -> Optional[ParsedResponse]:
        """Synchronous retrieve that uses the backend's event loop"""
        return self._run_coroutine(self.aretrieve(call_id, metadata=metadata))

    def __del__(self):
        """Clean up resources when the ConcurrentBackend is destroyed"""
        try:
            # Try to clean up the datastore from the concurrent thread first
            self.cleanup_datastore_sync()
            # Then shutdown the event loop
            self.shutdown()
        except Exception:
            # Ignore errors during cleanup in destructor
            pass
