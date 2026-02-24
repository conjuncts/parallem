from typing import TYPE_CHECKING

from parallellm.types import ProviderType


if TYPE_CHECKING:
    from parallellm.core.agent.orchestrator import AgentOrchestrator


class BatchNamespace:
    """Namespace for batch operations."""

    def __init__(self, orch: "AgentOrchestrator"):
        self._orch = orch

    def forget_batch(self, batch_uuid, *, provider_type=ProviderType) -> None:
        """
        Forget a batch of calls, removing them from the cache and preventing them from being used in future calls.

        :param batch_uuid: The batch UUID to forget
        :param provider_type: The provider type (e.g., 'openai', 'google')
        """
        # Get the datastore from the backend
        datastore = self._orch._backend._get_datastore()
        datastore.clear_batch_pending(batch_uuid)
