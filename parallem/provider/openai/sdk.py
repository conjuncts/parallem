from typing import TYPE_CHECKING, Optional, Union

from parallem.provider.openai.common import OpenAIBatchMixin
from parallem.provider.base import (
    AsyncProvider,
    BaseProvider,
    BatchProvider,
    SyncProvider,
)
from parallem.provider.openai.adapter import OpenAIAdapter
from parallem.types import (
    CommonQueryParameters,
    LLMIdentity,
    ParsedResponse,
)

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI
    from pydantic import BaseModel


class OpenAIProvider(BaseProvider):
    provider_type: str = "openai"

    def __init__(self):
        self.adapter = OpenAIAdapter()

    def validate_request_compatibility(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> None:
        """Validate OpenAI request compatibility.

        :param params: Common query parameters for the request.
        :return: None.
        """
        return None

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("gpt-5-nano", provider_type=self.provider_type)

    def parse_response(
        self, raw_response: Union["BaseModel", dict], llm: Optional[LLMIdentity] = None
    ) -> ParsedResponse:
        return self.adapter.convert_response(raw_response)


class SyncOpenAIProvider(SyncProvider, OpenAIProvider):
    def __init__(self, client: "OpenAI"):
        super().__init__()
        self.client = client

    def prepare_sync_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a synchronous callable for OpenAI API"""
        instructions = params["instructions"]
        fixed_documents = self.adapter.prepare_docs(params["strict_documents"])
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self.adapter.prepare_tools(params.get("tools"))

        if structured_output is not None:
            return self.client.responses.parse(
                model=llm.model_name,
                instructions=instructions,
                input=fixed_documents,
                text_format=structured_output,
                tools=tools,
                **kwargs,
            )

        return self.client.responses.create(
            model=llm.model_name,
            instructions=instructions,
            input=fixed_documents,
            tools=tools,
            **kwargs,
        )


class AsyncOpenAIProvider(AsyncProvider, OpenAIProvider):
    def __init__(self, client: "AsyncOpenAI"):
        super().__init__()
        self.client = client

    def prepare_async_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare an async coroutine for OpenAI API"""
        instructions = params["instructions"]
        fixed_documents = self.adapter.prepare_docs(params["strict_documents"])
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self.adapter.prepare_tools(params.get("tools"))

        if structured_output is not None:
            coro = self.client.responses.parse(
                model=llm.model_name,
                instructions=instructions,
                input=fixed_documents,
                text_format=structured_output,
                tools=tools,
                **kwargs,
            )
        else:
            coro = self.client.responses.create(
                model=llm.model_name,
                instructions=instructions,
                input=fixed_documents,
                tools=tools,
                **kwargs,
            )

        return coro


class BatchOpenAIProvider(OpenAIBatchMixin, BatchProvider, OpenAIProvider):
    batch_endpoint = "/v1/responses"

    def __init__(self, client: "OpenAI"):
        super().__init__()
        self.client = client

    def prepare_batch_call(
        self,
        params: CommonQueryParameters,
        custom_id: str,
        **kwargs,
    ) -> dict:
        """Prepare batch call data for OpenAI"""
        body = self.adapter.prepare_batch_request(params, **kwargs)
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
