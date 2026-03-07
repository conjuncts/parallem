from typing import TYPE_CHECKING

from pipelinellm.provider.base import BatchProvider
from pipelinellm.types import ProviderType


if TYPE_CHECKING:
    from pipelinellm.core.agent.orchestrator import AgentOrchestrator


class BatchNamespace:
    """Namespace for batch operations."""

    def __init__(self, orch: "AgentOrchestrator"):
        self._orch = orch

    def forget_batch(self, batch_uuid, *, provider_type: ProviderType) -> None:
        """
        Forget a batch of calls, removing them from the cache and preventing them from being used in future calls.

        :param batch_uuid: The batch UUID to forget
        :param provider_type: The provider type (e.g., 'openai', 'google')
        """
        # Get the datastore from the backend
        datastore = self._orch._backend._get_datastore()
        datastore.clear_batch_pending(batch_uuid)

    def _import_batch(
        self, batch_uuid, path_to_batch, *, provider_type: ProviderType
    ) -> None:
        """
        Not implemented yet - do not use.
        """
        # Note: fundamental limitation, because the batch is the _output_; there is no way to retrieve the
        # corresponding input (doc_hash) alone. Unless there is some way to guarantee the custom_id is the
        # doc_hash? I suppose that you should be allowed to manually import a file
        # if it has otherwise been run through and stored into SQLite. But then why do you need to do it
        # manually at all? This function is on hold.
        raise NotImplementedError

        if not self._orch._provider.is_compatible(provider_type):
            raise ValueError(
                f"Given provider '{provider_type}' is not compatible with current '{self._orch._provider.provider_type}'"
            )

        backend: "BatchProvider" = self._orch._provider
        with open(path_to_batch, "r") as f:
            content = f.read()
        batch_result = backend.decode_batch_content(content)
        datastore = self._orch._backend._get_datastore()

        # this requires the input to already be stored in the "pending"
        datastore.store_ready_batch(batch_result)
