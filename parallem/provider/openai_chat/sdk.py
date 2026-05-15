from typing import TYPE_CHECKING, Union

from parallem.provider.openai.common import OpenAIBatchMixin

from parallem.core.exception import ProviderCompatibilityError
from parallem.provider.base import (
    BaseProvider,
    BatchProvider,
    ConcurrentProvider,
    SyncProvider,
)
from parallem.provider.openai_chat.parser import OpenAIChatParser
from parallem.types import (
    CommonQueryParameters,
    LLMIdentity,
    ParsedResponse,
    ServerTool,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI
    from pydantic import BaseModel


class OpenAIChatProvider(BaseProvider):
    provider_type: str = "openai-chat"

    def __init__(self):
        super().__init__()
        self.parser = OpenAIChatParser()

    def validate_request_compatibility(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> None:
        """Validate OpenAI chat completions request compatibility.

        :param params: Common query parameters for the request.
        :return: None.
        """
        # web search tool is not supported in openai chat completions
        tools = params.get("tools") or []
        for tool in tools:
            if isinstance(tool, ServerTool) and tool.server_tool_type == "web_search":
                raise ProviderCompatibilityError(
                    "Web search tool is not supported with OpenAI ChatCompletions. Use Responses API instead."
                )
        return None

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("gpt-5-nano", provider_type=self.provider_type)

    def parse_response(
        self, raw_response: Union["BaseModel", dict], provider_type: str = None
    ) -> ParsedResponse:
        """Parse OpenAI chat completions response into common format."""
        return self.parser.convert_response(raw_response, provider_type=provider_type)

class SyncOpenAIChatProvider(SyncProvider, OpenAIChatProvider):
    def __init__(self, client: "OpenAI"):
        super().__init__()
        self.client = client

    def prepare_sync_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a synchronous callable for OpenAI chat completions API."""
        instructions = params["instructions"]
        fixed_documents = self.parser.fix_docs(
            params["strict_documents"], instructions
        )
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self.parser.fix_tools(params.get("tools"))

        if structured_output is not None:
            if "response_format" in kwargs:
                raise AssertionError(
                    "Cannot supply both structured_output and response_format"
                )
            kwargs["response_format"] = self.parser.fix_structured_output(structured_output)

        return self.client.chat.completions.create(
            model=llm.model_name,
            messages=fixed_documents,
            tools=tools,
            **kwargs,
        )


class ConcurrentOpenAIChatProvider(ConcurrentProvider, OpenAIChatProvider):
    def __init__(self, client: "AsyncOpenAI"):
        super().__init__()
        self.client = client

    def prepare_concurrent_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a concurrent coroutine for OpenAI chat completions API."""
        instructions = params["instructions"]
        fixed_documents = self.parser.fix_docs(
            params["strict_documents"], instructions
        )
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self.parser.fix_tools(params.get("tools"))

        if structured_output is not None:
            if "response_format" in kwargs:
                raise AssertionError(
                    "Cannot supply both structured_output and response_format"
                )
            kwargs["response_format"] = self.parser.fix_structured_output(structured_output)

        return self.client.chat.completions.create(
            model=llm.model_name,
            messages=fixed_documents,
            tools=tools,
            **kwargs,
        )


class BatchOpenAIChatProvider(OpenAIBatchMixin, BatchProvider, OpenAIChatProvider):
    batch_endpoint = "/v1/chat/completions"

    def __init__(self, client: "OpenAI"):
        super().__init__()
        self.client = client

    def prepare_batch_call(
        self,
        params: CommonQueryParameters,
        custom_id: str,
        **kwargs,
    ) -> dict:
        """Prepare batch call data for OpenAI chat completions."""
        body = self.parser.prepare_request(params, **kwargs)
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
