from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from parallem.types import (
    BatchIdentifier,
    BatchResult,
    BaseRetriever,
    CallIdentifier,
    ParsedError,
    ParsedResponse,
)

import polars as pl

if TYPE_CHECKING:
    from parallem.core.memoize.operations import OperationLog


class BaseDatastore(BaseRetriever, ABC):
    """
    Stores responses
    """

    @abstractmethod
    def persist(self) -> None:
        """
        Persist changes to file(s). Cleans up resources.
        """

    @abstractmethod
    def retrieve(
        self,
        call_id: CallIdentifier,
        metadata=False,
        *,
        origin_type: Optional[int] = None,
    ) -> Optional[ParsedResponse]:
        """
        Retrieve a response from the backend.

        :param call_id: The task identifier containing agent_name, doc_hash, and seq_id.
        :param metadata: Whether to include metadata in the response.
        :param origin_type: Optional origin marker filter. ``None`` retrieves only
            LLM-originated rows, ``1`` retrieves only human-originated rows.
        :returns: The retrieved response content.
        """

    @abstractmethod
    async def aretrieve(
        self,
        call_id: CallIdentifier,
        metadata=False,
        *,
        origin_type: Optional[int] = None,
    ) -> Optional[ParsedResponse]:
        """
        Asynchronously retrieve a response from the backend.

        :param call_id: The task identifier containing agent_name, doc_hash, and seq_id.
        :param metadata: Whether to include metadata in the response.
        :param origin_type: Optional origin marker filter. ``None`` retrieves only
            LLM-originated rows, ``1`` retrieves only human-originated rows.
        :returns: The retrieved response content.
        """

    @abstractmethod
    def populate_call_id(self, call_id: CallIdentifier) -> CallIdentifier:
        raise NotImplementedError

    @abstractmethod
    def store(
        self,
        call_id: CallIdentifier,
        parsed_response: ParsedResponse,
        *,
        displace=False,
        origin_type: Optional[int] = None,
    ) -> None:
        """
        Store a response in the backend.

        :param call_id: The task identifier containing doc_hash, seq_id, and session_id.
        :param parsed_response: The parsed response object containing text, response_id, and metadata.
        :param displace: If True, then invalidates any existing response with the same call_id
            before storing the new response (default: False).
        :param origin_type: Optional origin marker. ``None`` means LLM-originated,
            ``1`` means human-originated.
        """
        raise NotImplementedError

    # === begin error methods ===

    @abstractmethod
    def store_error(
        self,
        call_id: CallIdentifier,
        err: ParsedError,
    ) -> None:
        """
        Store an error response in the backend.

        :param call_id: The task identifier containing doc_hash, seq_id, and session_id.
        :param err: The error response object containing error details.
        """

    # === begin batch methods ===

    @abstractmethod
    def store_pending_batch(
        self,
        batch_id: BatchIdentifier,
    ) -> None:
        """
        Store pending batch information to track submitted batch requests.

        Should store call_ids, custom_ids, and batch_uuid so that when the batch completes,
        the results can be matched back to the original calls.

        :param batch_id: The batch identifier containing call_ids, custom_ids, and batch_uuid
        """

    @abstractmethod
    def store_ready_batch(
        self,
        batch_result: BatchResult,
        *,
        displace: bool = False,
    ) -> None:
        """
        Store completed batch results in the datastore.

        This takes the BatchResult, matches each response back to its original
        call_id using custom_id, and stores both the response and metadata.

        :param batch_result: The completed batch results to store
        :param displace: If True, displace existing records instead of inserting duplicates (default: False)
        """

    @abstractmethod
    def retrieve_batch_call_ids(self, batch_uuid: str) -> list[CallIdentifier]:
        """
        Retrieve all call_ids associated with an active batch_uuid.

        :param batch_uuid: The batch UUID to look up
        :returns: List of CallIdentifiers for this batch
        """

    @abstractmethod
    def get_all_pending_batch_uuids(self) -> list[tuple[str, str]]:
        """
        Retrieve all active pending batches from the datastore.

        :returns: List of tuples (batch_uuid, provider_type), one for each unique active pending batch
        """

    @abstractmethod
    def clear_batch_pending(self, batch_uuid: str) -> None:
        """
        Deactivate all pending batch records for a completed batch.

        :param batch_uuid: The batch UUID to deactivate
        """

    @abstractmethod
    def is_call_in_pending_batch(self, call_id: CallIdentifier) -> bool:
        """
        Check if a call_id is already in an active pending batch.

        :param call_id: The call identifier to check
        :returns: True if the call_id is in an active pending batch, False otherwise
        """

    @abstractmethod
    def export_polars(
        self,
    ) -> dict[str, "pl.DataFrame"]:
        """
        Export all tables from the datastore as Polars DataFrames.

        :returns: A dictionary mapping table names to Polars DataFrames.
        """
        raise NotImplementedError

    @abstractmethod
    def import_polars(
        self,
        tables: dict[str, "pl.DataFrame"],
        *,
        update: bool = True,
    ) -> None:
        """
        Set the datastore state from the provided Polars DataFrames.

        Each key in ``tables`` must match an existing table name.

        When ``update=True`` (default), rows are upserted via ``INSERT OR REPLACE``,
        so existing rows whose primary key matches are replaced in-place while rows
        with new primary keys are simply inserted. The rest of the table is left
        untouched.

        When ``update=False``, existing rows in each named table are deleted before
        inserting the new rows (full overwrite).

        :param tables: A dict mapping table names to Polars DataFrames.
        :param update: If True (default), upsert rows instead of overwriting the table.
        """

    # === begin memoize methods ===

    @abstractmethod
    def store_memoize(
        self,
        agent_name: str,
        state_hash: str,
        operation_log: "OperationLog",
    ) -> None:
        """
        Store memoized operation log for a given state hash.

        :param agent_name: The name of the agent owning the memoized state.
        :param state_hash: The hash of the initial MessageState.
        :param operation_log: The :class:`~parallem.core.memoize.operations.OperationLog`
            to persist.
        """

    @abstractmethod
    def retrieve_memoize(
        self,
        agent_name: str,
        state_hash: str,
    ) -> "Optional[OperationLog]":
        """
        Retrieve memoized operation log for a given state hash.

        :param agent_name: The name of the agent owning the memoized state.
        :param state_hash: The hash of the initial MessageState.
        :return: The :class:`~parallem.core.memoize.operations.OperationLog`, or
            ``None`` if not found.
        """
