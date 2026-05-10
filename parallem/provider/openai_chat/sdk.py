import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from parallem.core.exception import ProviderCompatibilityError
from parallem.provider.base import (
    BaseProvider,
    BatchProvider,
    ConcurrentProvider,
    SyncProvider,
)
from parallem.provider.openai.openai_tools import to_strict_json_schema
from parallem.types import (
    BatchResult,
    CommonQueryParameters,
    FunctionCall,
    FunctionCallOutput,
    FunctionCallRequest,
    LLMDocument,
    LLMIdentity,
    ParsedError,
    ParsedResponse,
    ServerTool,
)
from parallem.utils._batch_helper import _split_batch_response
from parallem.utils._quick_pydantic import is_pydantic_model
from parallem.utils.image import get_type_and_b64, is_image

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )
    from pydantic import BaseModel


class OpenAIChatProvider(BaseProvider):
    provider_type: str = "openai-chat"

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

    def _fix_docs_for_openai_chat(
        self,
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
                    "content": str(doc.content),
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
        self,
        tools: Optional[list[Union[dict, ServerTool]]],
    ) -> list[dict]:
        """Translate ServerTool into OpenAI chat completions format."""
        if tools is None:
            return []
        openai_tools = []
        for tool in tools:
            if isinstance(tool, ServerTool):
                if tool.server_tool_type == "web_search":
                    openai_tools.append({"type": "web_search", **tool.kwargs})
                elif tool.server_tool_type == "code_interpreter":
                    openai_tools.append({"type": "code_interpreter", **tool.kwargs})
                elif tool.server_tool_type == "mcp":
                    openai_tools.append(
                        {
                            "type": "mcp",
                            "server_label": tool.server_label,
                            "server_description": tool.server_description,
                            "server_url": tool.server_url,
                            "require_approval": tool.require_approval,
                        }
                    )
                else:
                    raise ValueError(
                        "Unsupported ServerTool type for OpenAI chat completions: "
                        f"{tool.server_tool_type}"
                    )
            elif isinstance(tool, dict):
                # If it's already in {"type": "function", "function": {...}} format, use as is
                if tool.get("type") == "function" and "function" in tool:
                    openai_tools.append(tool)
                elif "name" in tool and "parameters" in tool:
                    # Convert Responses-style schema to Chat Completions format
                    remainder = {k: v for k, v in tool.items() if k not in {"type"}}
                    openai_tools.append(
                        {
                            "type": "function",
                            "function": {
                                **remainder,
                            },
                        }
                    )
                else:
                    openai_tools.append(tool)
            else:
                openai_tools.append(tool)
        return openai_tools

    def _prepare_response_format(self, structured_output: object) -> dict:
        """Prepare chat completion response_format from structured_output input."""
        if isinstance(structured_output, dict):
            output_type = structured_output.get("type")
            if output_type in {"json_schema", "json_object"}:
                if (
                    output_type == "json_schema"
                    and "json_schema" not in structured_output
                ):
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

        raise ValueError(
            "structured_output must be a dict or a pydantic model for chat completions"
        )

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("gpt-5-nano", provider_type=self.provider_type)

    def parse_response(
        self, raw_response: Union["BaseModel", dict], provider_type: str = None
    ) -> ParsedResponse:
        """Parse OpenAI chat completions response into common format."""

        def _parse_choice(
            choice: dict, texts: list[str], calls: list[FunctionCall]
        ) -> None:
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

            resp_id = raw_response.pop("id", None)
            parsed_metadata = raw_response
        elif is_pydantic_model(raw_response):
            response: "ChatCompletion" = raw_response
            obj = response.model_dump(mode="json")
            resp_id = obj.pop("id", None)

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


class SyncOpenAIChatProvider(SyncProvider, OpenAIChatProvider):
    def __init__(self, client: "OpenAI"):
        self.client = client

    def prepare_sync_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a synchronous callable for OpenAI chat completions API."""
        instructions = params["instructions"]
        fixed_documents = self._fix_docs_for_openai_chat(
            params["strict_documents"], instructions
        )
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self._fix_tools_for_openai_chat(params.get("tools"))

        if structured_output is not None:
            if "response_format" in kwargs:
                raise AssertionError(
                    "Cannot supply both structured_output and response_format"
                )
            kwargs["response_format"] = self._prepare_response_format(structured_output)

        return self.client.chat.completions.create(
            model=llm.model_name,
            messages=fixed_documents,
            tools=tools,
            **kwargs,
        )


class ConcurrentOpenAIChatProvider(ConcurrentProvider, OpenAIChatProvider):
    def __init__(self, client: "AsyncOpenAI"):
        self.client = client

    def prepare_concurrent_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a concurrent coroutine for OpenAI chat completions API."""
        instructions = params["instructions"]
        fixed_documents = self._fix_docs_for_openai_chat(
            params["strict_documents"], instructions
        )
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self._fix_tools_for_openai_chat(params.get("tools"))

        if structured_output is not None:
            if "response_format" in kwargs:
                raise AssertionError(
                    "Cannot supply both structured_output and response_format"
                )
            kwargs["response_format"] = self._prepare_response_format(structured_output)

        return self.client.chat.completions.create(
            model=llm.model_name,
            messages=fixed_documents,
            tools=tools,
            **kwargs,
        )


class BatchOpenAIChatProvider(BatchProvider, OpenAIChatProvider):
    def __init__(self, client: "OpenAI"):
        self.client = client

    def prepare_batch_call(
        self,
        params: CommonQueryParameters,
        custom_id: str,
        **kwargs,
    ):
        """Prepare batch call data for OpenAI chat completions."""
        instructions = params["instructions"]
        fixed_documents = self._fix_docs_for_openai_chat(
            params["strict_documents"], instructions
        )
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self._fix_tools_for_openai_chat(params.get("tools"))

        if structured_output is not None:
            if "response_format" in kwargs:
                raise AssertionError(
                    "Cannot supply both structured_output and response_format"
                )
            kwargs["response_format"] = self._prepare_response_format(structured_output)

        body = {
            "model": llm.model_name,
            "messages": fixed_documents,
            "tools": tools,
            **kwargs,
        }
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    def _decode_openai_batch_result(self, result: dict) -> ParsedResponse:
        """Decode a single result from OpenAI batch response."""
        custom_id = result["custom_id"]
        body = result["response"]["body"]

        parsed = self.parse_response(body)
        parsed.custom_id = custom_id
        return parsed

    def _decode_openai_batch_error(self, result: dict) -> ParsedResponse:
        """Decode a single error from OpenAI batch response.

        The text field contains the error code.
        """
        custom_id = result.pop("custom_id", None)

        err_obj = result.get("error", {})
        resp_obj = result.get("response", {})
        resp_id = (resp_obj.get("body") or {}).get("id", None)
        error_code = resp_obj.get("status_code")
        if error_code is not None:
            error_code = str(error_code)

        if err_obj is not None:
            return ParsedResponse(
                text=error_code or "",
                response_id=resp_id,
                custom_id=custom_id,
                metadata=err_obj,
            )

        body = resp_obj.get("body", {})
        body_error = body.get("error", {})
        return ParsedResponse(
            text=error_code or "",
            response_id=resp_id,
            custom_id=custom_id,
            metadata=body_error,
        )

    def get_batch_custom_ids(self, stuff: list[dict], provider_type: str) -> list[str]:
        custom_ids = []
        for s in stuff:
            if not s.get("custom_id"):
                raise ValueError("Missing custom_id in batch item")
            custom_ids.append(s["custom_id"])
        return custom_ids

    def submit_batch_to_provider(self, fpath: Path, llm: LLMIdentity) -> str:
        """
        Submit a batch of calls to the provider.

        Returns the uuid

        This is called from the backend.
        """
        with open(fpath, "rb") as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")
            batch_input_file_id = batch_input_file.id

        batch_obj = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        return batch_obj.id

    def cancel_batch(self, batch_uuid: str, provider_type: str) -> None:
        """Cancel a batch on OpenAI."""
        self.client.batches.cancel(batch_uuid)

    def decode_batch_content(self, content: str) -> List[BatchResult]:
        """Decode content_str into dictionaries, with some error handling."""
        parsed_responses = []
        parsed_errors = []
        not_ok_i = []
        for line_i, line in enumerate(content.strip().split("\n")):
            if not line:
                continue
            try:
                line_data = json.loads(line)

                response = line_data.get("response", {})
                status_code = response.get("status_code")
                has_error = line_data.get("error") or (
                    status_code and status_code != 200
                )

                if not has_error and status_code == 200:
                    parsed_responses.append(self._decode_openai_batch_result(line_data))
                else:
                    decoded_err = self._decode_openai_batch_error(line_data)
                    decoded_err.error_code = status_code
                    parsed_errors.append(decoded_err)
                    not_ok_i.append(line_i)

            except json.JSONDecodeError as e:
                parsed_errors.append(
                    ParsedError(
                        text=f"JSON decode error: {str(e)}",
                        response_id=None,
                        custom_id="unknown",
                        metadata={},
                        error_code=1,
                    )
                )
                not_ok_i.append(line_i)

        return _split_batch_response(
            parsed_responses=parsed_responses,
            parsed_errors=parsed_errors,
            content=content,
            not_ok_i=not_ok_i,
        )

    def download_batch(
        self,
        batch_uuid: str,
        provider_type: str,
    ) -> List[BatchResult]:
        """Download the results of a batch from the provider.

        :param batch_uuid: The UUID of the batch to download.
        :return: List of BatchResult objects containing the results and errors (if any).
            If nothing is ready yet, empty list is returned.
        """
        batch = self.client.batches.retrieve(batch_uuid)
        err_file_id = batch.error_file_id
        out_file_id = batch.output_file_id

        if batch.errors:
            print(f"Batch {batch_uuid} failed with errors: {batch.errors}")
            return [
                BatchResult(
                    status="error",
                    raw_output=str(batch.errors),
                    parsed_responses=None,
                )
            ]

        if out_file_id is None and err_file_id is None:
            return []

        results = []

        if out_file_id is not None:
            out_content = self.client.files.content(out_file_id).text
            results.extend(self.decode_batch_content(out_content))

        if err_file_id is not None:
            err_content = self.client.files.content(err_file_id).text

            try:
                parsed_errors = [
                    self._decode_openai_batch_error(json.loads(line))
                    for line in err_content.strip().split("\n")
                    if line
                ]
                err_res = BatchResult(
                    status="error",
                    raw_output=err_content,
                    parsed_responses=parsed_errors,
                )
            except json.JSONDecodeError:
                err_res = BatchResult(
                    status="error",
                    raw_output=err_content,
                    parsed_responses=None,
                )
            results.append(err_res)
        return results
