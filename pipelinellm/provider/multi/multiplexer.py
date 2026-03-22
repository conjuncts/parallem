from pathlib import Path
from typing import Literal, overload

from pipelinellm.provider.base import (
    BaseProvider,
    BatchProvider,
    ConcurrentProvider,
    SyncProvider,
)
from pipelinellm.provider.multi.provider_selector import dynamic_select_provider
from pipelinellm.types import BatchResult, CommonQueryParameters, LLMIdentity


class MultiProvider(BaseProvider):
    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("gpt-5-nano", provider_type="openai")

    def is_compatible(self, other):
        return True

    def __init__(self, base_strategy: Literal["sync", "concurrent", "batch"]):
        self.providers: dict[str, BaseProvider] = {}
        self.base_strategy = base_strategy
        self.provider_type = None

    def validate_request_compatibility(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> None:
        """Validate request compatibility via the selected underlying provider.

        :param params: Common query parameters for the request.
        :return: None.
        """
        llm_identity = params["llm"]
        provider = self._load_provider(llm_identity.provider_type, self.base_strategy)
        provider.validate_request_compatibility(params, **kwargs)

    @overload
    def _load_provider(
        self, provider_name: str, strategy: Literal["sync"]
    ) -> SyncProvider: ...
    @overload
    def _load_provider(
        self, provider_name: str, strategy: Literal["concurrent"]
    ) -> ConcurrentProvider: ...
    @overload
    def _load_provider(
        self, provider_name: str, strategy: Literal["batch"]
    ) -> BatchProvider: ...
    def _load_provider(
        self, provider_name: str, strategy: Literal["sync", "concurrent", "batch"]
    ) -> BaseProvider:
        if provider_name is None:
            raise ValueError("Cannot resolve None provider.")
        if provider_name in self.providers:
            return self.providers[provider_name]
        else:
            provider = dynamic_select_provider(provider_name, strategy)
            self.providers[provider_name] = provider
            return provider

    def parse_response(self, raw_response, provider_type: str = None):
        provider = self._load_provider(provider_type, self.base_strategy)
        return provider.parse_response(raw_response)


class SyncMultiProvider(SyncProvider, MultiProvider):
    def __init__(self):
        super().__init__(base_strategy="sync")

    def prepare_sync_call(self, params: CommonQueryParameters, **kwargs):
        llm_identity = params["llm"]
        provider = self._load_provider(llm_identity.provider_type, "sync")
        return provider.prepare_sync_call(params, **kwargs)


class ConcurrentMultiProvider(ConcurrentProvider, MultiProvider):
    def __init__(self):
        super().__init__(base_strategy="concurrent")

    def prepare_concurrent_call(self, params: CommonQueryParameters, **kwargs):
        llm_identity = params["llm"]
        provider = self._load_provider(llm_identity.provider_type, "concurrent")
        return provider.prepare_concurrent_call(params, **kwargs)

    def parse_response(self, raw_response, provider_type: str = None):
        provider = self._load_provider(provider_type, "concurrent")
        return provider.parse_response(raw_response)


class BatchMultiProvider(BatchProvider, MultiProvider):
    def __init__(self):
        super().__init__(base_strategy="batch")

    def get_batch_custom_ids(self, stuff: list[dict], provider_type: str) -> list[str]:
        """Get batch IDs from a bunch of raw data."""
        provider = self._load_provider(provider_type, "batch")
        return provider.get_batch_custom_ids(stuff, provider_type=provider_type)

    def prepare_batch_call(self, params: CommonQueryParameters, **kwargs):
        llm_identity = params["llm"]
        provider = self._load_provider(llm_identity.provider_type, "batch")
        return provider.prepare_batch_call(params, **kwargs)

    def download_batch(
        self, batch_uuid: str, provider_type: str
    ) -> list["BatchResult"]:
        provider = self._load_provider(provider_type, "batch")
        return provider.download_batch(batch_uuid, provider_type=provider_type)

    def submit_batch_to_provider(self, fpath: Path, llm: LLMIdentity) -> str:
        provider = self._load_provider(llm.provider_type, "batch")
        return provider.submit_batch_to_provider(fpath, llm)

    def cancel_batch(self, batch_uuid: str, provider_type: str) -> None:
        provider = self._load_provider(provider_type, "batch")
        provider.cancel_batch(batch_uuid, provider_type=provider_type)

    def parse_response(self, raw_response, provider_type: str = None):
        provider = self._load_provider(provider_type, "batch")
        return provider.parse_response(raw_response, provider_type=provider_type)
