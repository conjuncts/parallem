from typing import TYPE_CHECKING, Optional
from parallem.provider.base import BaseProvider
from parallem.types import (
    BaseRetriever,
    CallIdentifier,
    CommonQueryParameters,
    ParsedResponse,
)

if TYPE_CHECKING:
    from parallem.core.datastore.base import BaseDatastore
    from parallem.core.datastore.input_storage import InputStorage


class BaseBackend(BaseRetriever):
    """
    A backend is a data store, but also a way to poll
    """

    def _get_datastore(self) -> "BaseDatastore":
        raise NotImplementedError

    def _get_input_storage(self) -> "InputStorage":
        raise NotImplementedError

    async def _poll_changes(self, call_id: CallIdentifier):
        """
        A chance to poll for changes and update the data store
        """
        raise NotImplementedError

    def persist(self):
        """Persist data and clean up resources"""
        pass

    def retrieve(
        self, call_id: CallIdentifier, metadata=False
    ) -> Optional[ParsedResponse]:
        """
        Retrieve a response.

        :param call_id: The task identifier containing agent_name, doc_hash, and seq_id.
        :returns: The retrieved ParsedResponse.
        """
        return self._get_datastore().retrieve(call_id, metadata=metadata)

    def populate_call_id(
        self, call_id: CallIdentifier, *, metadata=False
    ) -> CallIdentifier:
        return self._get_datastore().populate_call_id(call_id, metadata=metadata)

    def store_input(
        self,
        call_id: CallIdentifier,
        *,
        instructions: Optional[str],
        msgs: list,
        llm,
        structured_output,
        tools,
        hash_by,
        salt: Optional[str],
        request_kwargs: dict,
    ) -> None:
        self._get_input_storage().store_input(
            call_id,
            instructions=instructions,
            msgs=msgs,
            llm=llm,
            structured_output=structured_output,
            tools=tools,
            hash_by=hash_by,
            salt=salt,
            request_kwargs=request_kwargs,
        )

    def submit_query(
        self,
        provider: BaseProvider,
        params: CommonQueryParameters,
        *,
        call_id: CallIdentifier,
        **kwargs,
    ):
        raise NotImplementedError
