import time
from typing import Optional, TYPE_CHECKING
from parallem.core.backend import BaseBackend
from parallem.core.throttler import Throttler
from parallem.core.datastore.input_storage import InputStorage
from parallem.core.datastore.sqlite import SQLiteDatastore
from parallem.core.response import ReadyLLMResponse
from parallem.core.file_manager import FileManager
from parallem.logging.dash_logger import (
    DashboardLogger,
    HashStatus,
    PrimitiveDashboardLogger,
)
from parallem.types import (
    CallIdentifier,
    ParsedResponse,
    CommonQueryParameters,
)

if TYPE_CHECKING:
    from parallem.provider.base import SyncProvider


class SyncBackend(BaseBackend):
    """
    A synchronous backend that executes operations directly without threading or event loops.
    This backend is simpler and more straightforward for synchronous workflows.
    """

    def __init__(
        self,
        fm: FileManager,
        dashlog: DashboardLogger = PrimitiveDashboardLogger(),
        *,
        datastore_cls=None,
        rewrite_cache: bool = False,
        throttler=None,
    ):
        """
        Initialize the SyncBackend.

        :param fm: FileManager for data persistence
        :param dashlog: Optional dashboard logger for monitoring
        :param datastore_cls: Custom datastore class (defaults to SQLiteDatastore)
        :param rewrite_cache: Whether to overwrite existing cache entries
        :param throttler: Throttler instance for rate limiting (default: None)
        """
        self._fm = fm

        if datastore_cls is None:
            self._ds = SQLiteDatastore(self._fm)
        else:
            self._ds = datastore_cls(self._fm)
        self._input_storage = InputStorage(self._fm)
        self.dashlog = dashlog
        self._rewrite_cache = rewrite_cache

        if throttler is not None:
            self._throttler = throttler
        else:
            # No rate limiting by default
            self._throttler = Throttler(
                max_requests_per_window=None,
                window_seconds=None,
            )

    def _get_datastore(self):
        return self._ds

    def _get_input_storage(self):
        return self._input_storage

    def _apply_throttling(self) -> None:
        """Apply throttling by waiting if necessary"""
        delay = self._throttler.calculate_delay()
        if delay > 0:
            time.sleep(delay)
            # After waiting, record the actual submission time
            if self._throttler.is_enabled():
                self._throttler.record_request()

    def submit_query(
        self,
        provider: "SyncProvider",
        params: CommonQueryParameters,
        *,
        call_id: CallIdentifier,
        **kwargs,
    ) -> ReadyLLMResponse:
        """
        New control flow: Backend calls provider to get callable, then executes it.
        This inverts control from provider calling backend.
        """

        """Submit a synchronous function call and store the result immediately"""
        # Apply throttling before making the request
        self._apply_throttling()

        self.dashlog.update_call(call_id, HashStatus.SENT)

        provider.validate_request_compatibility(
            params,
            **kwargs,
        )

        # The below function typically calls the LLM
        result = provider.prepare_sync_call(
            params,
            **kwargs,
        )
        self.dashlog.update_call(call_id, HashStatus.RECEIVED)
        parsed = provider.parse_response(
            result, provider_type=params["llm"].provider_type
        )
        self._ds.store(call_id, parsed, upsert=self._rewrite_cache)

        return ReadyLLMResponse(call_id=call_id, pr=parsed)

    def _poll_changes(self, call_id: CallIdentifier):
        """
        Synchronous version - no polling needed since operations complete immediately
        """
        # In synchronous mode, all operations complete immediately
        # So there's nothing to poll for
        pass

    def retrieve(
        self, call_id: CallIdentifier, metadata=False
    ) -> Optional[ParsedResponse]:
        """
        Synchronous retrieve that checks datastore.

        The required fields from call_id are:
        - agent_name, doc_hash, seq_id
        """
        # Fall back to datastore
        return self._ds.retrieve(call_id, metadata=metadata)

    async def await_response(
        self, call_id: CallIdentifier, metadata: bool = False
    ) -> Optional[ParsedResponse]:
        return self.retrieve(call_id, metadata=metadata)

    def persist(self):
        """Persist any remaining data and datastore"""
        self._input_storage.persist()
        # Let datastore cleanup
        self._ds.persist()

    def close(self):
        """Clean up resources"""
        if hasattr(self._ds, "close"):
            self._ds.close()

    def __del__(self):
        """Clean up resources when the SyncBackend is destroyed"""
        try:
            self.close()
        except Exception:
            # Ignore errors during cleanup in destructor
            pass
