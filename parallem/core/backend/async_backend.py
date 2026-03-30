from typing import TYPE_CHECKING

from parallem.core.backend.concurrent_backend import ConcurrentBackend

if TYPE_CHECKING:
    from parallem.types import CallIdentifier, ParsedResponse


class AsyncBackend(ConcurrentBackend):
    """
    Async strategy backend.

    This backend reuses the concurrent execution engine but is intended to be
    consumed via ``await`` (for example: ``await llm_response``).
    """

    async def await_response(
        self, call_id: "CallIdentifier", metadata: bool = False
    ) -> "ParsedResponse | None":
        return await super().await_response(call_id, metadata=metadata)
