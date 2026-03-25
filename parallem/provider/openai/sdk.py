import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Union
from parallem.provider.base import (
    ConcurrentProvider,
    BaseProvider,
    BatchProvider,
    SyncProvider,
)
from parallem.provider.openai.openai_tools import to_strict_json_schema
from parallem.types import (
    BatchResult,
    CommonQueryParameters,
    FunctionCallRequest,
    FunctionCallOutput,
    LLMDocument,
    LLMIdentity,
    ParsedError,
    ParsedResponse,
    ServerTool,
    FunctionCall,
)
from parallem.utils._batch_helper import _split_batch_response
from parallem.utils._quick_pydantic import is_pydantic_model
from parallem.utils.image import get_type_and_b64, is_image

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI
    from openai.types.responses.response_input_param import Message
    from openai.types.responses.response import Response
    from pydantic import BaseModel


class OpenAIProvider(BaseProvider):
    provider_type: str = "openai"

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

    def _fix_docs_for_openai(
        self,
        documents: List[LLMDocument],
    ) -> "List[Message]":
        """Ensure documents are in the correct format for OpenAI API"""

        formatted_docs = []
        for doc in documents:
            if isinstance(doc, str):
                msg: "Message" = {
                    "role": "user",
                    "content": doc,
                }
                formatted_docs.append(msg)
            elif isinstance(doc, FunctionCallRequest):
                if doc.text_content:
                    formatted_docs.append(
                        {
                            "role": "assistant",
                            "content": doc.text_content,
                        }
                    )
                for call in doc.calls:
                    formatted_docs.append(
                        {
                            "name": call.name,
                            "arguments": call.arg_str,
                            "call_id": call.call_id,
                            "type": "function_call",
                        }
                    )
            elif isinstance(doc, FunctionCallOutput):
                msg = {
                    "type": "function_call_output",
                    "call_id": doc.call_id,
                    "output": str(doc.content),
                }
                formatted_docs.append(msg)
            elif isinstance(doc, tuple) and len(doc) == 2:
                # Handle Tuple[Literal["user", "assistant", "system", "developer"], str]
                # Valid roles for OpenAI are:
                # from openai.types.responses.response_input_param import Message
                # from openai.types.responses.response_output_message_param import ResponseOutputMessageParam
                role, content = doc

                msg: "Message" = {
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
                                "type": "input_image",
                                "image_url": f"data:{img_type};base64,{img_b64}",
                            },
                        ],
                    }
                )
            else:
                raise ValueError(f"Unsupported document type: {type(doc)}")
        return formatted_docs

    def _fix_server_tools_for_openai(
        self,
        tools: list[Union[dict, ServerTool]],
    ):
        """Translate ServerTool into OpenAI API format"""
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
                        f"Unsupported ServerTool type for OpenAI: {tool.server_tool_type}"
                    )
            else:
                openai_tools.append(tool)
        return openai_tools

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("gpt-5-nano", provider_type=self.provider_type)

    def parse_response(
        self, raw_response: Union["BaseModel", dict], provider_type: str = None
    ) -> ParsedResponse:
        """Parse OpenAI API response into common format"""
        if isinstance(raw_response, dict):
            # Dict response (e.g., from batch API)

            if "output_text" in raw_response:
                # Used in testing
                text = raw_response["output_text"]
                function_calls = []
            else:
                # Extract text from OpenAI responses API format
                function_calls = []
                texts: List[str] = []
                for output in raw_response.get("output", []):
                    if output["type"] == "function_call":
                        function_calls.append(
                            FunctionCall(
                                name=output["name"],
                                arguments=output["arguments"],
                                call_id=output.get("call_id"),
                            )
                        )
                    elif output["type"] == "custom_tool_call":
                        function_calls.append(
                            FunctionCall(
                                name=output["name"],
                                arguments=output["input"],
                                call_id=output.get("call_id"),
                            )
                        )
                    elif output["type"] == "message":
                        for content in output["content"]:
                            if content["type"] == "output_text":
                                texts.append(content["text"])
                text = "".join(texts)

            resp_id = raw_response.pop("id", None)
            parsed_metadata = raw_response
        elif is_pydantic_model(raw_response):
            # Pydantic model (e.g., from openai.types.responses.response.Response)
            response: Response = raw_response
            text = response.output_text
            obj = response.model_dump(mode="json")
            resp_id = response.id
            obj.pop("id", None)

            function_calls = []
            for item in response.output:
                if item.type == "function_call":
                    function_calls.append(
                        FunctionCall(
                            name=item.name,
                            arguments=item.arguments,
                            call_id=item.call_id,
                        )
                    )
                elif item.type == "custom_tool_call":
                    function_calls.append(
                        FunctionCall(
                            name=item.name, arguments=item.input, call_id=item.call_id
                        )
                    )

            parsed_metadata = obj
        else:
            raise ValueError(f"Unsupported response type: {type(raw_response)}")
        return ParsedResponse(
            text=text,
            response_id=resp_id,
            metadata=parsed_metadata,
            function_calls=function_calls,
        )


class SyncOpenAIProvider(SyncProvider, OpenAIProvider):
    def __init__(self, client: "OpenAI"):
        self.client = client

    def prepare_sync_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a synchronous callable for OpenAI API"""
        instructions = params["instructions"]
        fixed_documents = self._fix_docs_for_openai(params["strict_documents"])
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self._fix_server_tools_for_openai(params.get("tools"))

        if structured_output is not None:
            return self.client.responses.parse(
                model=llm.model_name,
                instructions=instructions,
                input=fixed_documents,
                text_format=structured_output,
                tools=tools,
                **kwargs,
            )

            # if "text" not in kwargs:
            #     kwargs["text"] = {}

            # schema = to_strict_json_schema(structured_output)
            # kwargs["text"]["format"] = {
            #     "type": "json_schema",
            #     "strict": True,
            #     "name": schema.get("title", "UnknownSchema"),
            #     "schema": schema,
            # }
        return self.client.responses.create(
            model=llm.model_name,
            instructions=instructions,
            input=fixed_documents,
            tools=tools,
            **kwargs,
        )


class ConcurrentOpenAIProvider(ConcurrentProvider, OpenAIProvider):
    def __init__(self, client: "AsyncOpenAI"):
        self.client = client

    def prepare_concurrent_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a concurrent coroutine for OpenAI API"""
        instructions = params["instructions"]
        fixed_documents = self._fix_docs_for_openai(params["strict_documents"])
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self._fix_server_tools_for_openai(params.get("tools"))

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


class BatchOpenAIProvider(BatchProvider, OpenAIProvider):
    def __init__(self, client: "OpenAI"):
        self.client = client

    def prepare_batch_call(
        self,
        params: CommonQueryParameters,
        custom_id: str,
        **kwargs,
    ):
        """Prepare batch call data for OpenAI"""
        instructions = params["instructions"]
        fixed_documents = self._fix_docs_for_openai(params["strict_documents"])
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self._fix_server_tools_for_openai(params.get("tools"))

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

    def _decode_openai_batch_result(self, result: dict) -> ParsedResponse:
        """Decode a single result from OpenAI batch response"""
        custom_id = result["custom_id"]

        body = result["response"]["body"]

        # destroyed metadata:
        # result["error"] (should be null)
        # result["response"]["status_code"] (should be 200)
        # redundant IDs (supplanted by custom_id):
        #   result["id"]
        #   result["response"]["request_id"]
        #   result["response"]["body"]["id"]

        parsed = self.parse_response(body)
        parsed.custom_id = custom_id
        return parsed

    def _decode_openai_batch_error(self, result: dict) -> ParsedResponse:
        """Decode a single error from OpenAI batch response

        The text field contains the error code.
        """
        custom_id = result.pop("custom_id", None)
        # destroyed metadata:
        # redundant IDs (see above, supplanted by custom_id)
        # result["response"]["body"]

        err_obj = result.get("error", {})
        # actually, error_obj is usually None, and the real error is in response.body.error

        resp_obj = result.get("response", {})
        resp_id = (resp_obj.get("body") or {}).get("id", None)
        error_code = resp_obj.get("status_code")
        if error_code is not None:
            error_code = str(error_code)

        if err_obj is not None:
            # unusually, the error is indeed here
            return ParsedResponse(
                text=error_code or "",
                response_id=resp_id,
                custom_id=custom_id,
                metadata=err_obj,
            )

        # need to retrieve error from response body
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
            endpoint="/v1/responses",
            completion_window="24h",
            # metadata={"description":""}
        )

        return batch_obj.id

    def cancel_batch(self, batch_uuid: str, provider_type: str) -> None:
        """Cancel a batch on OpenAI."""
        self.client.batches.cancel(batch_uuid)

    def decode_batch_content(self, content: str) -> List[BatchResult]:
        """Decode content_str into dictionaries, with some error handling"""
        parsed_responses = []
        parsed_errors = []
        not_ok_i = []
        for line_i, line in enumerate(content.strip().split("\n")):
            if not line:
                continue
            try:
                line_data = json.loads(line)
                # custom_id = line_data.get("custom_id", "unknown")

                # Check if it's a successful response (status_code 200) or an error
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
                # Handle malformed JSON lines
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
            return [BatchResult(
                status="error",
                raw_output=str(batch.errors),
                parsed_responses=None,
            )]

        if out_file_id is None and err_file_id is None:
            return []

        results = []

        # Successful completion
        out_content = self.client.files.content(out_file_id).text
        results.extend(self.decode_batch_content(out_content))

        # Errors
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
