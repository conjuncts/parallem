from typing import TYPE_CHECKING, List, Optional
from parallem.provider.base import BaseProvider
from parallem.types import (
    BaseRetriever,
    CallIdentifier,
    CommonQueryParameters,
    HashByOption,
    LLMResponse,
    ParsedResponse,
)

if TYPE_CHECKING:
    from parallem.core.datastore.base import BaseDatastore


class BaseBackend(BaseRetriever):
    """
    A backend is a data store, but also a way to poll
    """

    def _get_datastore(self) -> "BaseDatastore":
        raise NotImplementedError

    async def _poll_changes(self, call_id: CallIdentifier):
        """
        A chance to poll for changes and update the data store
        """
        raise NotImplementedError

    def persist(self):
        """Persist data and clean up resources"""
        pass

    def retrieve(self, call_id: CallIdentifier, metadata=False) -> Optional[ParsedResponse]:
        """
        Retrieve a response.

        :param call_id: The task identifier containing agent_name, doc_hash, and seq_id.
        :returns: The retrieved ParsedResponse.
        """
        return self._get_datastore().retrieve(call_id, metadata=metadata)

    def call_llm(
        self,
        provider: BaseProvider,
        params: CommonQueryParameters,
        *,
        call_id: CallIdentifier,
        **kwargs,
    ) -> LLMResponse:
        raise NotImplementedError
