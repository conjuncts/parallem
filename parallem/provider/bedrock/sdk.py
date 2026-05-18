import asyncio
import json
from typing import TYPE_CHECKING, Optional, Union

from parallem.core.exception import ProviderCompatibilityError
from parallem.provider.base import (
    BaseProvider,
    ConcurrentProvider,
    SyncProvider,
)
from parallem.provider.bedrock.adapter import BedrockAdapter
from parallem.types import (
    CommonQueryParameters,
    LLMIdentity,
    ParsedResponse,
)

if TYPE_CHECKING:
    from pydantic import BaseModel
    try:
        from mypy_boto3_bedrock import BedrockClient
    except ImportError:
        from botocore.client import BaseClient as BedrockClient


class BedrockProvider(BaseProvider):
    provider_type: str = "bedrock"

    def __init__(self, client: "BedrockClient"):
        super().__init__()
        self.adapter = BedrockAdapter()
        self.client = client

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")

    def validate_request_compatibility(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> None:
        """Validate Bedrock InvokeModel compatibility.

        :param params: Common query parameters for the request.
        :return: None.
        """
        if params.get("tools"):
            raise ProviderCompatibilityError(
                "Bedrock InvokeModel does not support tools in this provider."
            )
        if params.get("structured_output") is not None:
            raise ProviderCompatibilityError(
                "Structured output is not supported for Bedrock InvokeModel."
            )

    def parse_response(
        self, raw_response: Union["BaseModel", dict], llm: Optional[LLMIdentity] = None
    ) -> ParsedResponse:
        """Parse Bedrock InvokeModel response into common format.

        :param raw_response: The raw response from Bedrock.
        :param provider_type: Optional provider override.
        :return: ParsedResponse instance.
        """
        return self.adapter.convert_response(raw_response)


class SyncBedrockProvider(SyncProvider, BedrockProvider):

    def prepare_sync_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a synchronous Bedrock InvokeModel call.

        :param params: Common query parameters for the request.
        :return: Raw response from Bedrock.
        """
        llm = params["llm"]
        model_kwargs = kwargs.copy()
        model_name, body, invoke_options = self.adapter.fix_config(params, model_kwargs)

        if not isinstance(body, (str, bytes)):
            body = json.dumps(body)

        content_type = invoke_options.pop("contentType", "application/json")
        accept = invoke_options.pop("accept", "application/json")

        return self.client.invoke_model(
            modelId=llm.model_name,
            body=body,
            contentType=content_type,
            accept=accept,
            **invoke_options,
        )


class ConcurrentBedrockProvider(ConcurrentProvider, BedrockProvider):

    def prepare_concurrent_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a concurrent coroutine for Bedrock InvokeModel.

        :param params: Common query parameters for the request.
        :return: Coroutine yielding a raw response.
        """

        async def _runner():
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: SyncBedrockProvider(self.client).prepare_sync_call(
                    params, **kwargs
                ),
            )

        return _runner()
