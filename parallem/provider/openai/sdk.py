from typing import TYPE_CHECKING, Union

from parallem.provider.openai.common import OpenAIBatchMixin
from parallem.provider.base import (
    ConcurrentProvider,
    BaseProvider,
    BatchProvider,
    SyncProvider,
)
from parallem.provider.openai.openai_tools import to_strict_json_schema
from parallem.provider.openai.parser import OpenAIParser
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
        self.parser = OpenAIParser()

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
        self, raw_response: Union["BaseModel", dict], provider_type: str = None
    ) -> ParsedResponse:
        return self.parser.convert_response(raw_response, provider_type=provider_type)


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
        fixed_documents = self.parser.fix_docs(params["strict_documents"])
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self.parser.fix_tools(params.get("tools"))

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


class ConcurrentOpenAIProvider(ConcurrentProvider, OpenAIProvider):
    def __init__(self, client: "AsyncOpenAI"):
        super().__init__()
        self.client = client

    def prepare_concurrent_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a concurrent coroutine for OpenAI API"""
        instructions = params["instructions"]
        fixed_documents = self.parser.fix_docs(params["strict_documents"])
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self.parser.fix_tools(params.get("tools"))

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
    ):
        """Prepare batch call data for OpenAI"""
        instructions = params["instructions"]
        fixed_documents = self.parser.fix_docs(params["strict_documents"])
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self.parser.fix_tools(params.get("tools"))

        if structured_output is not None:
            if "text" not in kwargs:
                kwargs["text"] = {}

            assert not kwargs["text"].get("format"), (
                "Cannot supply both structured_output and text.format"
            )
            schema = to_strict_json_schema(structured_output)
            kwargs["text"]["format"] = {
                "type": "json_schema",
                "strict": True,
                "name": schema.get("title", "UnknownSchema"),
                "schema": schema,
            }

        body = {
            "model": llm.model_name,
            "instructions": instructions,
            "input": fixed_documents,
            "tools": tools,
            **kwargs,
        }
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
