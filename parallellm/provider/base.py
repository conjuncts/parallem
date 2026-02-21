from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union
from pydantic import BaseModel

from parallellm.types import (
    BatchIdentifier,
    BatchResult,
    CallIdentifier,
    CommonQueryParameters,
    LLMIdentity,
    ProviderType,
    ParsedResponse,
)

if TYPE_CHECKING:
    from parallellm.file_io.file_manager import FileManager


class BaseProvider:
    provider_type: Optional[ProviderType] = None
    """Must be set by subclasses to identify the provider type."""

    def is_compatible(self, other: ProviderType) -> bool:
        """Returns whether this provider accepts the given provider type."""
        return other is None or self.provider_type == other

    def get_default_llm_identity(self) -> LLMIdentity:
        """Returns a default LLMIdentity for this provider."""
        raise NotImplementedError

    def parse_response(self, raw_response: Union[BaseModel, dict]) -> ParsedResponse:
        """
        Parse a raw API response into a common format.

        :param raw_response: The raw response from the API (Pydantic model or dict)
        :return: ParsedResponse containing response_text, response_id, and metadata_dict
        """
        raise NotImplementedError


class SyncProvider(BaseProvider):
    def prepare_sync_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """
        Eagerly makes the API call.

        :param params: Common query parameters containing instructions, documents, llm, etc.
        :return: Raw response from the API call
        """
        raise NotImplementedError


class ConcurrentProvider(BaseProvider):
    def prepare_concurrent_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """
        Prepare a concurrent coroutine for the backend to execute.

        :param params: Common query parameters containing instructions, documents, llm, etc.
        :return: A coroutine that when awaited will make the API call and return the raw response
        """
        raise NotImplementedError


class BatchProvider(BaseProvider):
    def prepare_batch_call(
        self,
        params: CommonQueryParameters,
        custom_id: str,
        **kwargs,
    ):
        """
        Prepare batch call data for the backend to bookkeep.

        :param params: Common query parameters containing instructions, documents, llm, etc.
        :return: A dict/object representing the batch request format for this provider
        """
        raise NotImplementedError

    def get_batch_custom_ids(self, stuff: list[dict]) -> list[str]:
        """Get batch IDs from a bunch of raw data."""
        raise NotImplementedError

    def submit_batch_to_provider(self, fpath: Path, llm: str) -> str:
        """Submit a batch of calls to the provider."""
        raise NotImplementedError

    def download_batch(self, batch_uuid: str) -> List[BatchResult]:
        """Download the results of a batch from the provider.

        The list can contain both ready and error results.
        Empty list = still pending.
        - batch_status is one of "pending", "ready", or "error".
        """
        raise NotImplementedError
