from collections.abc import Awaitable
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

if TYPE_CHECKING:
    from pydantic import BaseModel

from parallem.types import (
    BatchResult,
    CommonQueryParameters,
    LLMDocument,
    LLMIdentity,
    ProviderType,
    ParsedResponse,
    ServerTool,
)


class BaseAdapter:
    """Helps adapt API inputs."""

    def fix_config(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> Any:
        """Make config ready for API calls. Also fixes documents and tools."""
        raise NotImplementedError

    def fix_docs(
        self,
        documents: List[LLMDocument],
    ):
        """Make documents ready for API calls."""
        raise NotImplementedError

    def fix_tools(
        self,
        tools: List[Union[dict, ServerTool]],
    ):
        """Make tools ready for API calls."""
        raise NotImplementedError

    def prepare_request(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> dict:
        """Prepare full request payload."""
        raise NotImplementedError

    def convert_response(self, raw_response: Union["BaseModel", dict]) -> ParsedResponse:
        """Parse raw API response into common format."""
        raise NotImplementedError


class BaseProvider:
    provider_type: Optional[ProviderType] = None
    """Must be set by subclasses to identify the provider type."""

    def is_compatible(self, other: ProviderType) -> bool:
        """Returns whether this provider accepts the given provider type."""
        return other is None or self.provider_type == other

    def get_default_llm_identity(self) -> LLMIdentity:
        """Returns a default LLMIdentity for this provider."""
        raise NotImplementedError

    def validate_request_compatibility(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> None:
        """Validate whether a request is compatible with this provider.

        :param params: Common query parameters for the request.
        :return: None. Raises when the request is not compatible.
        """
        raise NotImplementedError

    def parse_response(
        self, raw_response: Union["BaseModel", dict], llm: Optional[LLMIdentity] = None
    ) -> ParsedResponse:
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


class AsyncProvider(BaseProvider):
    def prepare_async_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> Awaitable[Any]:
        """
        Prepare an async coroutine for the backend to execute.

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
    ) -> dict:
        """
        Prepare batch call data for the backend to bookkeep.

        :param params: Common query parameters containing instructions, documents, llm, etc.
        :return: A dict/object representing the batch request format for this provider
        """
        raise NotImplementedError

    def get_batch_custom_ids(self, stuff: list[dict], provider_type: str) -> list[str]:
        """Get batch IDs from dicts."""
        raise NotImplementedError

    def submit_batch_to_provider(self, fpath: Path, llm: LLMIdentity) -> str:
        """Submit a batch of calls to the provider."""
        raise NotImplementedError

    def download_batch(self, batch_uuid: str, provider_type: str) -> List[BatchResult]:
        """Download the results of a batch from the provider.

        The list can contain both ready and error results.
        Empty list = still pending.
        - batch_status is one of "pending", "ready", or "error".

        :param provider_type: Double check to make sure that batch_uuid is for the same provider.
        :param batch_uuid: The unique identifier for the batch to download
        """
        raise NotImplementedError

    def cancel_batch(self, batch_uuid: str, provider_type: str) -> None:
        """Cancel a batch on the provider.

        :param provider_type: Double check to make sure that batch_uuid is for the same provider.
        :param batch_uuid: The unique identifier for the batch to cancel.
        :return: None.
        """
        raise NotImplementedError
