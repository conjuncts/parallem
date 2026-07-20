import base64
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Union, overload

from parallem.provider.base import (
    AsyncProvider,
    BaseAdapter,
    BaseProvider,
    BatchProvider,
    SyncProvider,
)
from parallem.types import (
    BatchResult,
    CommonQueryParameters,
    FileInput,
    FunctionCall,
    FunctionCallOutput,
    FunctionCallRequest,
    LLMDocument,
    LLMIdentity,
    MCPOutput,
    MultipartDocument,
    ParsedError,
    ParsedResponse,
    ServerTool,
)
from parallem.utils._batch_helper import _split_batch_response
from parallem.utils._quick_pydantic import is_pydantic_model
from parallem.utils.image import get_type_and_b64, is_image, is_image_url
from parallem.utils.manip import maybe_snake_to_camel

if TYPE_CHECKING:
    from google import genai
    from google.genai import types
    from mcp.types import ContentBlock
    from pydantic import BaseModel


def _fix_part_for_google(
    part: LLMDocument,
    *,
    strict: bool = True
) -> "types.PartDict":
    """Ensure a single part is in the correct format for Gemini API

    :param strict:
        If True, always return a types.PartDict.
        If False, allow returning string (which the SDK accepts)
    """
    if isinstance(part, str):
        if not strict:
            return part
        return {"text": part}
    elif is_image(part):
        img_type, img_b64 = get_type_and_b64(
            part,
            allowed=[
                "image/png",
                "image/jpeg",
                "image/webp",
                "image/heic",
                "image/heif",
            ],
        )
        return {
            "inline_data": {
                "mime_type": img_type,
                "data": img_b64,
            }
        }
    elif is_image_url(part):
        return {
            "file_data": {
                "mime_type": part.mime_type,
                "file_uri": str(part),
            }
        }
    elif isinstance(part, FileInput):
        if part.file_content:
            return {
                "inline_data": {
                    "mime_type": part.mime_type,
                    "data": base64.b64encode(part.file_content).decode("utf-8"),
                }
            }
        elif part.file_url:
            return {
                "file_data": {
                    "mime_type": part.mime_type,
                    "file_uri": part.file_url,
                }
            }
        else:
            raise ValueError("Could not handle FileInput for Google Gemini")
    elif isinstance(part, dict):
        return part
    else:
        raise ValueError(f"Unsupported part type for Google Gemini: {type(part)}")


@overload
def _fix_docs_for_google(
    documents: list[LLMDocument],
    strict: Literal[True] = True,
) -> list["types.ContentDict"]:
    ...

@overload
def _fix_docs_for_google(
    documents: list[LLMDocument],
    strict: Literal[False],
) -> list["types.ContentDict | types.PartDict"]:
    ...

def _fix_docs_for_google(
    documents: list[LLMDocument],
    strict: bool = True,
) -> list[Union["types.ContentDict", "types.PartDict"]]:
    ...
    """Ensure documents are in the correct format for Gemini API.

    Note that the API allows a mixture of content and parts (types.ContentListUnionDict)
    :param documents: List of LLMDocument (str, tuple, dict, FileInput, FunctionCallRequest, FunctionCallOutput)
    :param strict: If True, always return types.ContentDict.
        If False, allow returning types.PartDict or str, to more faithfully match
        the Gemini Python SDK.
    """

    formatted_docs: list[Union[str, "types.ContentDict"]] = []
    for doc in documents:
        if isinstance(doc, str):
            part = _fix_part_for_google(doc, strict=strict)
            if strict:
                formatted_docs.append({
                    "role": "user",
                    "parts": [part]
                })
            else:
                formatted_docs.append(part)
        elif isinstance(doc, (FunctionCallOutput, MCPOutput)):
            if isinstance(doc, MCPOutput):
                response_content = {
                    "output": [_fix_mcp_block(block) for block in doc.content],
                }
            else:
                response_content = {"output": doc.content}

            function_response_part: "types.PartDict" = {
                "function_response": {
                    "name": doc.name,
                    "response": response_content,
                }
            }
            formatted_docs.append(
                {
                    "role": "user",
                    "parts": [function_response_part],
                }
            )
        elif isinstance(doc, FunctionCallRequest):
            parts: list["types.PartDict"] = []
            if doc.text_content:
                parts.append({"text": doc.text_content})
            for call in doc.calls:
                parts.append(
                    {
                        "function_call": {
                            "name": call.name,
                            "args": call.args,
                            "id": call.fcall_id,
                        }
                    }
                )
            formatted_docs.append(
                {
                    "role": "model",
                    "parts": parts,
                }
            )
        elif isinstance(doc, tuple) and len(doc) == 2:
            role, content = doc
            if role == "assistant":
                role = "model"
            elif role != "user":
                raise ValueError(f"Unsupported role for Google: {role}")
            formatted_docs.append(
                {
                    "role": role,
                    "parts": [{"text": content}],
                }
            )
        elif (
            isinstance(doc, FileInput)
            or is_image(doc)
            or is_image_url(doc)
        ):
            msg: "types.PartDict" = _fix_part_for_google(doc, strict=strict)
            if strict:
                formatted_docs.append(
                    {
                        "role": "user",
                        "parts": [msg],
                    }
                )
            else:
                formatted_docs.append(msg)
        elif isinstance(doc, dict):
            formatted_docs.append(doc)
        elif isinstance(doc, MultipartDocument):
            formatted_docs.append({
                "role": doc.role,
                "parts": [_fix_part_for_google(part, strict=strict) for part in doc.parts],
            })
        else:
            raise ValueError(f"Unsupported document type: {type(doc)}")

    return formatted_docs


def _fix_mcp_block(content_block: "ContentBlock") -> dict:
    if content_block.type == "text":
        return {
            "type": "text",
            "text": content_block.text,
        }
    if content_block.type == "image":
        return {
            "type": "image",
            "mime_type": content_block.mimeType,
            "data": content_block.data,
        }
    if content_block.type == "resource":
        resource = content_block.resource
        output = {"type": "resource"}

        try:
            blob = resource.blob
        except AttributeError:
            blob = None
        if blob is not None:
            output["blob"] = blob

        try:
            text = resource.text
        except AttributeError:
            text = None
        if text is not None:
            output["text"] = text

        try:
            uri = resource.uri
        except AttributeError:
            uri = None
        if uri is not None:
            output["uri"] = str(uri)

        try:
            mime_type = resource.mimeType
        except AttributeError:
            mime_type = None
        if mime_type is not None:
            output["mime_type"] = mime_type

        return output

    return {
        "type": "text",
        "text": content_block.model_dump_json(exclude_none=True),
    }


def _prepare_tool_schema(
    func_schemas: List[Union[dict, ServerTool]],
) -> List["types.ToolDict"]:
    """Convert tool definitions to Google Tool schema"""

    google_tools: list["types.ToolDict"] = []
    for sch in func_schemas:
        if isinstance(sch, ServerTool):
            extra_params: Union["types.GoogleSearchDict", "types.ToolCodeExecutionDict"] = (
                sch.kwargs or {}
            )
            if sch.server_tool_type == "web_search":
                google_tools.append({"google_search": extra_params})
            elif sch.server_tool_type == "code_interpreter":
                google_tools.append({"code_execution": extra_params})
            elif sch.server_tool_type == "mcp":
                obj = {
                    "type": "mcp_server",
                    "name": sch.server_label,
                    "url": sch.server_url,
                }
                google_tools.append(obj)
            else:
                raise ValueError(
                    f"Unsupported ServerTool type for Google Gemini: {sch.server_tool_type}"
                )
        elif isinstance(sch, Mapping):
            if "type" in sch:
                sch = sch.copy()
                sch.pop("type")
            function_tool: "types.ToolDict" = {"function_declarations": [sch]}
            google_tools.append(function_tool)
        else:
            google_tools.append(sch)
    return google_tools


def _extract_text_from_gemini_model(resp: "BaseModel"):
    """Extract text content from Gemini API response."""
    if (
        not resp.candidates
        or not resp.candidates[0].content
        or not resp.candidates[0].content.parts
    ):
        return None
    text = ""
    any_text_part_text = False
    for part in resp.candidates[0].content.parts:
        if isinstance(part.text, str):
            if isinstance(part.thought, bool) and part.thought:
                continue
            any_text_part_text = True
            text += part.text
    return text if any_text_part_text else None


def _extract_text_from_gemini_dict(resp: dict):
    """Extract text content from Gemini API response."""
    if (
        not resp.get("candidates")
        or not resp["candidates"][0].get("content")
        or not resp["candidates"][0]["content"].get("parts")
    ):
        return None
    text = ""
    any_text_part_text = False
    for part in resp["candidates"][0]["content"]["parts"]:
        if isinstance(part.get("text"), str):
            if isinstance(part.get("thought"), bool) and part["thought"]:
                continue
            any_text_part_text = True
            text += part["text"]
    return text if any_text_part_text else None


def _camel_case_items(items: dict) -> dict:
    """Convert keys in a dictionary to camelCase."""
    if isinstance(items, list):
        return [_camel_case_items(item) for item in items]
    if not isinstance(items, dict):
        return items

    result = {}
    for k, v in items.items():
        if k == "properties":
            result[k] = {prop_k: _camel_case_items(prop_v) for prop_k, prop_v in v.items()}
        else:
            result[maybe_snake_to_camel(k)] = _camel_case_items(v)

    return result


def _capitalize_function_decl(items: dict) -> None:
    """Convert function declaration types to CAPITAL, because enums must be capitalized.
    MUTATES the object."""
    if "type" in items:
        items["type"] = items["type"].upper()
    if "properties" in items:
        for k, v in items["properties"].items():
            _capitalize_function_decl(v)


class GoogleAdapter(BaseAdapter):
    def prepare_sdk_request(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> tuple[str, List["types.ContentDict"], "types.GenerateContentConfigDict"]:
        instructions = params["instructions"]
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = params.get("tools")

        contents = self.prepare_docs(params["strict_documents"])

        config: "types.GenerateContentConfigDict" = kwargs.copy()
        if instructions:
            config["system_instruction"] = instructions

        if structured_output is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = structured_output

        if tools:
            config["tools"] = self.prepare_tools(tools)

        model_name = llm.model_name if llm else "gemini-2.5-flash"

        return model_name, contents, config

    def prepare_docs(
        self,
        documents: List[LLMDocument],
    ):
        """Make documents ready for API calls."""
        return _fix_docs_for_google(documents)

    def prepare_tools(
        self,
        tools: List[Union[dict, ServerTool]],
    ):
        """Make tools ready for API calls."""
        return _prepare_tool_schema(tools)

    def prepare_batch_request(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> dict:
        """Prepare full request payload."""
        model_name, fixed_documents, config = self.prepare_sdk_request(params, **kwargs)

        big_config = {}
        for k in [
            "tools",
            "tool_config",
            "safety_settings",
            "system_instruction",
            "cached_content",
            "service_tier",
            "store",
        ]:
            if k in config:
                big_config[maybe_snake_to_camel(k)] = _camel_case_items(config.pop(k))

        if "systemInstruction" in big_config and isinstance(big_config["systemInstruction"], str):
            big_config["systemInstruction"] = {
                "parts": [{"text": big_config["systemInstruction"]}],
            }

        gen_config = _camel_case_items(config.pop("generation_config", {}))
        if "response_schema" in config:
            sch = config.pop("response_schema")
            if not isinstance(sch, dict):
                if getattr(sch, "model_json_schema", None):
                    sch = sch.model_json_schema()
            gen_config["responseJsonSchema"] = sch
        gen_config.update(_camel_case_items(config))
        if gen_config:
            big_config["generationConfig"] = gen_config

        for tool in big_config.get("tools") or []:
            for decl in tool.get("functionDeclarations") or []:
                if "parameters" in decl:
                    _capitalize_function_decl(decl["parameters"])

        body = {
            "contents": _camel_case_items(fixed_documents),
            **big_config,
        }
        return body

    def convert_response(self, raw_response: Union["BaseModel", dict]) -> ParsedResponse:
        """Parse Gemini API response into common format"""
        if isinstance(raw_response, dict):
            resp_id = raw_response.get("response_id") or raw_response.get("responseId")

            text_content = _extract_text_from_gemini_dict(raw_response)

            tools = []
            for part in raw_response.get("candidates", [{}])[0].get("content", {}).get("parts", []):
                func_call = None
                if "functionCall" in part:
                    func_call = part["functionCall"]
                elif "function_call" in part:
                    func_call = part["function_call"]

                if func_call:
                    tools.append(
                        FunctionCall(
                            name=func_call.get("name"),
                            arguments=func_call.get("args"),
                            fcall_id=func_call.get("id"),
                        )
                    )

            if text_content is None:
                text_content = ""
            return ParsedResponse(
                text=text_content,
                response_id=resp_id,
                metadata=raw_response,
                function_calls=tools,
            )
        elif is_pydantic_model(raw_response):
            response: "types.GenerateContentResponse" = raw_response
            text = _extract_text_from_gemini_model(response)
            response_id = response.response_id
            obj = response.model_dump(mode="json")

            tools = []

            if not response.candidates:
                pass

            for part in response.candidates[0].content.parts or []:
                if part.function_call:
                    func_call: "types.FunctionCall" = part.function_call
                    tools.append(
                        FunctionCall(
                            name=func_call.name,
                            arguments=func_call.args,
                            fcall_id=func_call.id,
                        )
                    )
            if text is None:
                text = ""
            return ParsedResponse(
                text=text, response_id=response_id, metadata=obj, function_calls=tools
            )
        else:
            raise ValueError(f"Unsupported response type: {type(raw_response)}")


class GoogleProvider(BaseProvider):
    provider_type: str = "google"

    def __init__(self):
        self.adapter = GoogleAdapter()

    def validate_request_compatibility(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> None:
        """Validate Google request compatibility.

        :param params: Common query parameters for the request.
        :return: None.
        """
        return None

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("gemini-2.5-flash", provider_type=self.provider_type)

    def parse_response(
        self, raw_response: Union["BaseModel", dict], llm: Optional[LLMIdentity] = None
    ) -> ParsedResponse:
        """Parse Gemini API response into common format"""
        return self.adapter.convert_response(raw_response)


class SyncGoogleProvider(SyncProvider, GoogleProvider):
    def __init__(self, client: "genai.Client"):
        super().__init__()
        self.client = client

    def prepare_sync_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a synchronous callable for Gemini API"""
        model_name, contents, config = self.adapter.prepare_sdk_request(params, **kwargs)
        return self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )


class AsyncGoogleProvider(AsyncProvider, GoogleProvider):
    def __init__(self, client: "genai.Client"):
        super().__init__()
        self.client = client

    def prepare_async_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare an async coroutine for Gemini API"""
        model_name, contents, config = self.adapter.prepare_sdk_request(params, **kwargs)

        coro = self.client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        return coro


class BatchGoogleProvider(BatchProvider, GoogleProvider):
    def __init__(self, client: "genai.Client"):
        super().__init__()
        self.client = client

    def prepare_batch_call(
        self,
        params: CommonQueryParameters,
        custom_id: str,
        **kwargs,
    ) -> dict:
        """Convert CommonQueryParameters to Gemini batch request format"""
        body = self.adapter.prepare_batch_request(params, **kwargs)
        request = {
            "key": custom_id,
            "request": body,
        }

        return request

    def _decode_gemini_batch_result(self, result: dict, custom_id: str) -> ParsedResponse:
        """Decode a single result from Gemini batch response"""

        # For successful responses, the result should be a GenerateContentResponse
        if "response" in result:
            response_data = result["response"]
            # Use existing parse_response method to handle the response
            parsed = self.parse_response(response_data)
            parsed.custom_id = custom_id
            return parsed
        else:
            # Fallback parsing
            raise ValueError("Unexpected gemini response format")

    def _decode_gemini_batch_error(self, result: dict, custom_id: str) -> ParsedResponse:
        """Decode a single error from Gemini batch response"""

        error_info = result.get("error", {})
        error_message = error_info.get("message", "Unknown error")
        response_id = result.get("response_id", None)

        return ParsedResponse(
            text=error_message,
            response_id=response_id,
            custom_id=custom_id,
            metadata=error_info,
            error_code=1,  # TODO: figure out exact format of gemini
        )

    def get_batch_custom_ids(self, stuff, provider_type: str):
        custom_ids = []
        for item in stuff:
            if "key" not in item:
                raise ValueError("Each batch item must have a 'key' field.")
            custom_ids.append(item["key"])
        return custom_ids

    def submit_batch_to_provider(self, fpath: Path, llm: LLMIdentity) -> str:
        """
        Submit a batch of calls to the provider.

        This is called from the backend.
        """
        from google.genai import types

        num_lines = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for _ in f:
                num_lines += 1
        # Upload the file to Gemini File API
        uploaded_file = self.client.files.upload(
            file=fpath,
            config=types.UploadFileConfig(
                display_name=f"batch-requests-{num_lines}",
                mime_type="application/json",
            ),
        )

        # Create batch job
        batch_job = self.client.batches.create(
            model=llm.model_name,
            src=uploaded_file.name,
            config={
                "display_name": f"batch-job-{num_lines}-requests",
            },
        )
        return batch_job.name.removeprefix("batches/")

        # TODO: consider: a try/finally block option to clean up the temp files?

    def cancel_batch(self, batch_uuid: str, provider_type: str) -> None:
        """Cancel a Gemini batch job."""
        self.client.batches.cancel(name="batches/" + batch_uuid)

    def decode_batch_content(
        self,
        content: str,
    ) -> List[BatchResult]:
        """Decode content_str into dictionaries, with some error handling"""
        parsed_responses = []
        parsed_errors = []
        not_ok_i = []
        for line_i, line in enumerate(content.strip().split("\n")):
            if line:
                try:
                    line_data = json.loads(line)
                    custom_id = line_data.get("key", "unknown")

                    if "response" in line_data:
                        parsed_responses.append(
                            self._decode_gemini_batch_result(line_data, custom_id)
                        )
                    else:
                        parsed_errors.append(self._decode_gemini_batch_error(line_data, custom_id))
                        not_ok_i.append(line_i)

                except json.JSONDecodeError as e:
                    # TODO: Handle malformed JSON lines
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

        # Get batch job status
        batch_job = self.client.batches.get(name="batches/" + batch_uuid)

        # Check if job is completed
        completed_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }

        if batch_job.state.name not in completed_states:
            return []  # Still pending

        results = []

        if batch_job.state.name == "JOB_STATE_SUCCEEDED":
            if batch_job.dest and batch_job.dest.file_name:
                # Results are in a file
                try:
                    file_content = self.client.files.download(file=batch_job.dest.file_name)
                    content_str = file_content.decode("utf-8")

                    results.extend(self.decode_batch_content(content_str))
                except Exception as e:
                    results.append(
                        BatchResult(
                            status="error",
                            raw_output=str(e),
                            parsed_responses=None,
                        )
                    )
            else:
                raise ValueError("Batch job succeeded but no destination file found.")

        else:
            # Job failed, cancelled, or expired
            error_msg = getattr(batch_job, "error", f"Job state: {batch_job.state.name}")
            results.append(
                BatchResult(
                    status="error",
                    raw_output=str(error_msg),
                    parsed_responses=None,
                )
            )

        return results
