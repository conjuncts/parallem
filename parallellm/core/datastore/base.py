from abc import ABC
from typing import List, Optional, Union

from parallellm.types import (
    BatchIdentifier,
    BatchResult,
    CallIdentifier,
    LLMDocument,
    LLMResponse,
    ParsedError,
    ParsedResponse,
)


class Datastore(ABC):
    """
    Stores responses
    """

    def persist(self) -> None:
        """
        Persist changes to file(s). Cleans up resources.
        """
        raise NotImplementedError

    def retrieve(
        self, call_id: CallIdentifier, metadata=False
    ) -> Optional[ParsedResponse]:
        """
        Retrieve a response from the backend.

        :param call_id: The task identifier containing agent_name, doc_hash, and seq_id.
        :param metadata: Whether to include metadata in the response.
        :returns: The retrieved response content.
        """
        raise NotImplementedError

    def retrieve_metadata_legacy(self, response_id: str) -> Optional[dict]:
        """
        Retrieve metadata from the backend using response_id.

        :param response_id: The response ID to look up metadata for.
        :returns: The retrieved metadata as a dictionary, or None if not found.
        """
        raise NotImplementedError

    def store(
        self,
        call_id: CallIdentifier,
        parsed_response: ParsedResponse,
        *,
        upsert=False,
    ) -> None:
        """
        Store a response in the backend.

        :param call_id: The task identifier containing doc_hash, seq_id, and session_id.
        :param parsed_response: The parsed response object containing text, response_id, and metadata.
        :param upsert: If True, update existing record instead of inserting duplicate (default: False)
        """
        raise NotImplementedError

    # === begin error methods ===

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
        raise NotImplementedError

    # === begin SaveInput methods ===

    def store_input(
        self,
        doc_hash: str,
        *,
        instructions: Optional[str],
        msgs: List[Union[LLMDocument, LLMResponse]],
        msg_hashes: list[str],
        salt_terms: list[str],
    ):
        pass

    # === begin batch methods ===

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
        raise NotImplementedError

    def store_ready_batch(
        self,
        batch_result: BatchResult,
        *,
        upsert: bool = False,
    ) -> None:
        """
        Store completed batch results in the datastore.

        This takes the BatchResult, matches each response back to its original
        call_id using custom_id, and stores both the response and metadata.

        :param batch_result: The completed batch results to store
        :param upsert: If True, update existing records instead of inserting duplicates (default: False)
        """
        raise NotImplementedError

    def retrieve_batch_call_ids(self, batch_uuid: str) -> list[CallIdentifier]:
        """
        Retrieve all call_ids associated with an active batch_uuid.

        :param batch_uuid: The batch UUID to look up
        :returns: List of CallIdentifiers for this batch
        """
        raise NotImplementedError

    def get_all_pending_batch_uuids(self) -> list[tuple[str, str]]:
        """
        Retrieve all active pending batches from the datastore.

        :returns: List of tuples (batch_uuid, provider_type), one for each unique active pending batch
        """
        raise NotImplementedError

    def clear_batch_pending(self, batch_uuid: str) -> None:
        """
        Deactivate all pending batch records for a completed batch.

        :param batch_uuid: The batch UUID to deactivate
        """
        raise NotImplementedError

    def is_call_in_pending_batch(self, call_id: CallIdentifier) -> bool:
        """
        Check if a call_id is already in an active pending batch.

        :param call_id: The call identifier to check
        :returns: True if the call_id is in an active pending batch, False otherwise
        """
        raise NotImplementedError
