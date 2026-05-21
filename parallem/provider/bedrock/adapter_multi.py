import json
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
        
        geo_prefix = None
        model_name = llm.model_name
        _key = None
        # remove geographical prefix
        first_period = model_name.find(".")
        if 2 <= first_period <= 4:
            # assume xx., xxx., xxxx. are geographical prefixes
            geo_prefix, model_name = model_name.split(".", 1)

        # cannot just split by period. for example: gpt-3.5-turbo
        if model_name.startswith("amazon."):
            _key = "amazon"
        elif model_name.startswith("anthropic."):
            _key = "anthropic"
        else:
            # most models accept OpenAI ChatCompletions
            # "moonshotai.",
            # "deepseek.",
            # "google.",
            # "openai.",
            # "qwen.",
            # "minimax.",
            # "mistral.", - most models
            # "nvidia.",
            # "zai.",
            _key = "openai-chat"

        if _key not in self._adapters:
            if _key == "amazon":
                from parallem.provider.bedrock.adapter_nova import BedrockNovaAdapter
                self._adapters[_key] = BedrockNovaAdapter()
            elif _key == "openai-chat":
                from parallem.provider.openai_chat.adapter import OpenAIChatAdapter
                self._adapters[_key] = OpenAIChatAdapter()
            elif _key == "anthropic":
                from parallem.provider.bedrock.adapter_anthropic import BedrockAnthropicAdapter
                self._adapters[_key] = BedrockAnthropicAdapter()
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

        # Should be of format:
        # {"ResponseMetadata": {"RequestId": "abc123"}, "body": {...}}
        if not isinstance(raw_response, dict):
            raise ValueError(f"Unsupported Bedrock response type: {type(raw_response)}")

        body_payload = raw_response.get("body")
        if hasattr(body_payload, "read"):
            raw_body = body_payload.read()
        else:
            # unexpected
            raw_body = body_payload or raw_response

        if isinstance(raw_body, dict):
            body_obj = raw_body
        else:
            if isinstance(raw_body, bytes):
                raw_body_text = raw_body.decode("utf-8")
            else:
                raw_body_text = raw_body or ""

            try:
                body_obj = json.loads(raw_body_text) if raw_body_text else {}
            except json.JSONDecodeError:
                body_obj = {}

        adapter = self._get_adapter_for(llm) if llm is not None else None
        if adapter is not None:
            return adapter.convert_response(body_obj)
            