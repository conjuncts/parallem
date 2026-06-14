from typing import TYPE_CHECKING, List, Optional, Union

from parallem.provider.base import BaseAdapter
from parallem.provider.openai.common import map_server_tools

from parallem.provider.openai.openai_tools import to_strict_json_schema
from parallem.types import (
    CommonQueryParameters,
    FunctionCall,
    FunctionCallOutput,
    FunctionCallRequest,
    LLMDocument,
    ParsedResponse,
    ServerTool,
)
from parallem.utils._quick_pydantic import is_pydantic_model
from parallem.utils.image import get_type_and_b64, is_image

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )
    from pydantic import BaseModel


def _fix_docs_for_openai_chat(
    documents: List[LLMDocument],
    instructions: Optional[str],
) -> "List[ChatCompletionMessageParam]":
    """Ensure documents are in the correct format for OpenAI chat completions."""
    formatted_docs: list[dict] = []
    if instructions:
        formatted_docs.append(
            {
                "role": "system",
                "content": instructions,
            }
        )

    for doc in documents:
        if isinstance(doc, str):
            msg: "ChatCompletionMessageParam" = {
                "role": "user",
                "content": doc,
            }
            formatted_docs.append(msg)
        elif isinstance(doc, FunctionCallRequest):
            msg: dict = {
                "role": "assistant",
                "content": doc.text_content if doc.text_content else None,
                "tool_calls": [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arg_str,
                        },
                    }
                    for call in doc.calls
                ],
            }
            formatted_docs.append(msg)
        elif isinstance(doc, FunctionCallOutput):
            msg = {
                "role": "tool",
                "tool_call_id": doc.call_id,
                "content": doc.content,
            }
            formatted_docs.append(msg)
        elif isinstance(doc, tuple) and len(doc) == 2:
            role, content = doc
            if role == "developer":
                role = "system"
            msg = {
                "role": role,
                "content": content,
            }
            formatted_docs.append(msg)
        elif is_image(doc):
            img_type, img_b64 = get_type_and_b64(
                doc, allowed=["image/jpeg", "image/png", "image/gif", "image/webp"]
            )
            formatted_docs.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{img_type};base64,{img_b64}",
                            },
                        }
                    ],
                }
            )
        else:
            raise ValueError(f"Unsupported document type: {type(doc)}")
    return formatted_docs


def _fix_tools_for_openai_chat(
    tools: Optional[list[Union[dict, ServerTool]]],
) -> list[dict]:
    """Translate ServerTool into OpenAI chat completions format."""
    openai_tools = map_server_tools(tools, web_search_supported=False)
    chat_tools = []
    for tool in openai_tools:
        if isinstance(tool, dict):
            # ChatCompletions expects {"type": "function", "function": {...}}
            # whereas Responses expects {"type": "function", "name": ..., "parameters": ...}
            if tool.get("type") == "function" and "function" in tool:
                chat_tools.append(tool)
            elif "name" in tool and "parameters" in tool:
                remainder = {k: v for k, v in tool.items() if k not in {"type"}}
                chat_tools.append(
                    {
                        "type": "function",
                        "function": {
                            **remainder,
                        },
                    }
                )
            else:
                chat_tools.append(tool)
        else:
            chat_tools.append(tool)
    return chat_tools


def _prepare_response_format(structured_output: object) -> dict:
    """Prepare chat completion response_format from structured_output input."""
    if isinstance(structured_output, dict):
        output_type = structured_output.get("type")
        if output_type in {"json_schema", "json_object"}:
            if output_type == "json_schema" and "json_schema" not in structured_output:
                schema_obj = structured_output.get("schema", structured_output)
                return {
                    "type": "json_schema",
                    "json_schema": {
                        "name": structured_output.get("name", "UnknownSchema"),
                        "schema": schema_obj,
                        "strict": structured_output.get("strict", True),
                    },
                }
            return structured_output
        if "response_format" in structured_output:
            return structured_output["response_format"]
        return {
            "type": "json_schema",
            "json_schema": {
                "name": structured_output.get("title", "UnknownSchema"),
                "schema": structured_output,
                "strict": True,
            },
        }

    model_json_schema = getattr(structured_output, "model_json_schema", None)
    if callable(model_json_schema):
        schema = to_strict_json_schema(structured_output)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.get("title", "UnknownSchema"),
                "schema": schema,
                "strict": True,
            },
        }

    raise ValueError("structured_output must be a dict or a pydantic model for chat completions")


class OpenAIChatAdapter(BaseAdapter):
    def fix_config(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        raise NotImplementedError

    def fix_docs(
        self,
        documents: List[LLMDocument],
        instructions: Optional[str] = None,
    ) -> "List[ChatCompletionMessageParam]":
        return _fix_docs_for_openai_chat(documents, instructions)

    def fix_tools(
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

    def prepare_request(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> dict:
        """Prepare OpenAI API request parameters from common query parameters."""
        instructions = params["instructions"]
        fixed_documents = self.fix_docs(params["strict_documents"], instructions)
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self.fix_tools(params.get("tools"))

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
        """Parse OpenAI API response into common format."""

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
                            call_id=tool_call.get("id"),
                        )
                    )

            legacy_call = message.get("function_call")
            if legacy_call:
                calls.append(
                    FunctionCall(
                        name=legacy_call.get("name"),
                        arguments=legacy_call.get("arguments"),
                        call_id=None,
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
