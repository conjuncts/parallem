from typing import TYPE_CHECKING, List, Optional, Union

from parallem.provider.base import BaseAdapter
from parallem.provider.bedrock.adapter_legacy import BedrockLegacyAdapter

from parallem.types import CommonQueryParameters, LLMDocument, LLMIdentity, ParsedResponse, ServerTool


if TYPE_CHECKING:
    from pydantic import BaseModel

class BedrockAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self._fallback_adapter = BedrockLegacyAdapter()
        self._adapters = {}
        
    def _get_adapter_for(
        self,
        llm: LLMIdentity,
    ):
        if any(
            llm.model_name.startswith(prefix) for prefix in ["amazon.", "us.amazon.", "eu.amazon.", "ap.amazon.", "cn.amazon."]
        ):
            _key = "amazon"
        elif llm.model_name.startswith("openai."):
            _key = "openai"
        elif llm.model_name.startswith("anthropic."):
            _key = "anthropic"
        else:
            _key = "google"
        
        if _key not in self._adapters:
            if _key == "amazon":
                from parallem.provider.bedrock.adapter_nova import BedrockNovaAdapter
                self._adapters[_key] = BedrockNovaAdapter()
            elif _key == "openai":
                from parallem.provider.openai.adapter import OpenAIAdapter
                self._adapters[_key] = OpenAIAdapter()
            elif _key == "anthropic":
                from parallem.provider.anthropic.adapter import AnthropicAdapter
                self._adapters[_key] = AnthropicAdapter()
            elif _key == "google":
                from parallem.provider.google.adapter import GoogleAdapter
                self._adapters[_key] = GoogleAdapter()
        return self._adapters[_key]

    def fix_config(
        self,
        params: CommonQueryParameters,
        **kwargs: dict,
    ):
        """Convert common query parameters into Bedrock-specific config."""
        llm = params["llm"]
        adapter = self._get_adapter_for(llm)
        return adapter.fix_config(params, **kwargs)

    def fix_docs(
        self,
        documents: List[LLMDocument],
        *,
        llm: LLMIdentity = None,
    ):
        assert llm is not None, "LLMIdentity must be provided to fix_docs for BedrockAdapter"
        adapter = self._get_adapter_for(llm)
        return adapter.fix_docs(documents)

    def fix_tools(
        self,
        tools: List[Union[dict, ServerTool]],
        llm: LLMIdentity = None,
    ):
        """Make tools ready for API calls."""
        assert llm is not None, "LLMIdentity must be provided to fix_tools for BedrockAdapter"
        adapter = self._get_adapter_for(llm)
        return adapter.fix_tools(tools)

    def prepare_request(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> dict:
        """Prepare full request payload."""
        llm = params["llm"]
        adapter = self._get_adapter_for(llm)
        return adapter.prepare_request(params, **kwargs)

    def convert_response(
        self, raw_response: Union["BaseModel", dict], llm: Optional[LLMIdentity] = None
    ) -> ParsedResponse:
        """Parse raw API response into common format."""
        assert llm is not None, "LLMIdentity must be provided to convert_response for BedrockAdapter"

        adapter = self._get_adapter_for(llm) if llm is not None else None
        if adapter is not None:
            return adapter.convert_response(raw_response)
            