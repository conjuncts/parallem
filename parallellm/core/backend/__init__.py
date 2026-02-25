from typing import TYPE_CHECKING, List, Optional, Union
from parallellm.provider.base import BaseProvider
from parallellm.types import (
    CallIdentifier,
    CommonQueryParameters,
    LLMDocument,
    ParsedResponse,
)

if TYPE_CHECKING:
    from parallellm.core.datastore.base import Datastore


class BaseBackend:
    """
    A backend is a data store, but also a way to poll
    """

    def _get_datastore(self) -> "Datastore":
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
        raise NotImplementedError

    def submit_query(
        self,
        provider: BaseProvider,
        params: CommonQueryParameters,
        *,
        call_id: CallIdentifier,
        **kwargs,
    ):
        raise NotImplementedError
