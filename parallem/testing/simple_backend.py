from typing import Optional
from parallem.core.backend import BaseBackend
from parallem.core.backend.async_backend import AsyncBackend
from parallem.core.backend.sync_backend import SyncBackend
from parallem.core.calls import _call_to_concise_dict
from parallem.core.datastore.base import BaseDatastore
from parallem.types import CallIdentifier, ParsedResponse


class MockBackend(BaseBackend):
    """
    A simple in-memory backend for testing purposes.
    """

    def __init__(self):
        self._dict = {}

    async def _poll_changes(self, call_id: CallIdentifier):
        pass

    def persist(self):
        pass

    def retrieve(
        self,
        call_id: CallIdentifier,
        metadata=False,
        *,
        origin_type: Optional[int] = None,
    ) -> Optional[ParsedResponse]:
        """
        Retrieve a response.

        :param call_id: The task identifier containing agent_name, doc_hash, and seq_id.
        :returns: The retrieved ParsedResponse.
        """
        return self._dict.get(tuple(_call_to_concise_dict(call_id).values()))

    def store(
        self,
        call_id: CallIdentifier,
        response: ParsedResponse,
        *,
        displace: bool = False,
        origin_type: Optional[int] = None,
    ):
        self._dict[tuple(_call_to_concise_dict(call_id).values())] = response


class MockDatastore(BaseDatastore):
    """
    Simple in-memory datastore for testing purposes.
    """

    def __init__(self, fm=None):
        self._dict = {}

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
        :returns: The retrieved response content.
        """
        return self._dict.get(tuple(_call_to_concise_dict(call_id).values()))

    def store(
        self,
        call_id: CallIdentifier,
        parsed_response: ParsedResponse,
        *,
        displace: bool = False,
        origin_type: Optional[int] = None,
    ):
        """
        Store a response in the backend.

        :param call_id: The task identifier containing doc_hash, and seq_id.
        :param parsed_response: The parsed response object containing text, response_id, and metadata.
        """
        self._dict[tuple(_call_to_concise_dict(call_id).values())] = parsed_response

    def persist(self) -> None:
        """
        Persist changes to file(s). Cleans up resources.
        """
        pass


class MockSyncBackend(SyncBackend):
    def __init__(self):
        super().__init__(None, None, datastore_cls=MockDatastore)


class MockAsyncBackend(AsyncBackend):
    def __init__(self):
        super().__init__(None, datastore_cls=MockDatastore)


# class MockBatchBackend(BatchBackend):
#     def __init__(self):
#         super().__init__(None, None, datastore_cls=MockDatastore)
