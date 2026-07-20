from typing import TYPE_CHECKING, List, Optional, Union

from parallem.provider.base import BaseAdapter
from parallem.provider.openai_chat.sdk import (
    _fix_docs_for_openai_chat,
    _fix_tools_for_openai_chat,
    _prepare_response_format,
)
from parallem.types import (
    CommonQueryParameters,
    FunctionCall,
    LLMDocument,
    ParsedResponse,
    ServerTool,
)
from parallem.utils._quick_pydantic import is_pydantic_model

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )
    from pydantic import BaseModel


class MistralAdapter(BaseAdapter):
    """Adapter for Mistral AI API (OpenAI-compatible chat completions).

    Mistral's API is compatible with OpenAI's chat completions format, so this
    adapter reuses the OpenAI chat completions document and tool formatting.
    """

    def prepare_sdk_request(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        raise NotImplementedError(
            "MistralAdapter does not implement prepare_sdk_request; "
            "use prepare_docs/prepare_tools directly instead."
        )

    def prepare_docs(
        self,
        documents: List[LLMDocument],
        instructions: Optional[str] = None,
    ) -> "List[ChatCompletionMessageParam]":
        return _fix_docs_for_openai_chat(documents, instructions)

    def prepare_tools(
        self,
        tools: List[Union[dict, ServerTool]],
    ):
        """Make tools ready for API calls."""
        return _fix_tools_for_openai_chat(tools)

    def fix_structured_output(
        self,
        structured_output: object,
    ) -> dict:
        return _prepare_response_format(structured_output)

    def prepare_batch_request(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> dict:
        """Prepare Mistral API request parameters from common query parameters."""
        instructions = params["instructions"]
        fixed_documents = self.prepare_docs(params["strict_documents"], instructions)
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self.prepare_tools(params.get("tools"))

        if structured_output is not None:
            if "response_format" in kwargs:
                raise AssertionError("Cannot supply both structured_output and response_format")
            kwargs["response_format"] = self.fix_structured_output(structured_output)

        body = {
            "model": llm.model_name,
            "messages": fixed_documents,
            "tools": tools,
            **kwargs,
        }
        return body

    def convert_response(
        self,
        raw_response: Union["BaseModel", dict],
    ) -> ParsedResponse:
        """Parse Mistral API response into common format."""

        def _parse_choice(choice: dict, texts: list[str], calls: list[FunctionCall]) -> None:
            message = choice.get("message") or {}
            content = message.get("content")
            if content:
                texts.append(content)

            tool_calls = message.get("tool_calls") or []
            for tool_call in tool_calls:
                if tool_call.get("type") == "function":
                    function_obj = tool_call.get("function") or {}
                    calls.append(
                        FunctionCall(
                            name=function_obj.get("name"),
                            arguments=function_obj.get("arguments"),
                            fcall_id=tool_call.get("id"),
                        )
                    )

        if isinstance(raw_response, dict):
            choices = raw_response.get("choices") or []
            texts: list[str] = []
            function_calls: list[FunctionCall] = []
            for choice in choices:
                _parse_choice(choice, texts, function_calls)
            text = "".join(texts)

            resp_id = raw_response.get("id")
            parsed_metadata = raw_response
        elif is_pydantic_model(raw_response):
            response: "ChatCompletion" = raw_response
            obj = response.model_dump(mode="json")
            resp_id = obj.get("id")

            choices = obj.get("choices") or []
            texts = []
            function_calls = []
            for choice in choices:
                _parse_choice(choice, texts, function_calls)
            text = "".join(texts)

            parsed_metadata = obj
        else:
            raise ValueError(f"Unsupported response type: {type(raw_response)}")

        return ParsedResponse(
            text=text,
            response_id=resp_id,
            metadata=parsed_metadata,
            function_calls=function_calls,
        )
