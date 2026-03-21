import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Union
from pipelinellm.provider.base import (
    ConcurrentProvider,
    BaseProvider,
    BatchProvider,
    SyncProvider,
)
from pipelinellm.types import (
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
from pipelinellm.utils._quick_pydantic import is_pydantic_model

if TYPE_CHECKING:
    from google import genai
    from google.genai import types
    from pydantic import BaseModel

from pipelinellm.utils._batch_helper import _split_batch_response
from pipelinellm.utils.image import get_type_and_b64, is_image
from pipelinellm.utils.manip import maybe_snake_to_camel


def _fix_docs_for_google(
    documents: List[LLMDocument],
) -> List["types.ContentDict"]:
    """Ensure documents are in the correct format for Gemini API"""

    # For Gemini, we can pass strings directly or convert to proper format
    # The SDK will handle the conversion automatically
    # if len(documents) == 1 and isinstance(documents[0], str):
    # return documents[0]  # Single string can be passed directly

    # For multiple documents or mixed types, return as list
    formatted_docs: list[Union[str, "types.ContentDict"]] = []
    for doc in documents:
        if isinstance(doc, str):
            formatted_docs.append(
                {
                    "role": "user",
                    "parts": [{"text": doc}],
                }
            )
        elif isinstance(doc, FunctionCallOutput):
            # https://ai.google.dev/gemini-api/docs/function-calling?example=meeting
            function_response_part: "types.PartDict" = {
                "function_response": {
                    "name": doc.name,
                    "response": {"output": str(doc.content)},
                }
            }
            # types.Part()
            # types.FunctionResponse()
            # types.Content()
            formatted_docs.append(
                {
                    "parts": [function_response_part],
                    "role": "user",
                }
            )

        elif isinstance(doc, FunctionCallRequest):
            parts: list["types.PartDict"] = []
            if doc.text_content:
                parts.append({"text": doc.text_content})
                # parts.append(types.Part(text=types.Text(content=doc.text_content)))
            for call in doc.calls:
                parts.append(
                    {
                        "function_call": {
                            "name": call.name,
                            "args": call.args,
                            "id": call.call_id,
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
            # from google.genai.types import Content
            # For google, only valid roles are ["user", "model"]

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
        elif isinstance(doc, dict):
            # If it's already a proper content dict, keep it
            if "parts" in doc and "role" in doc:
                formatted_docs.append(doc)
            else:
                raise ValueError(f"Invalid document dict format for Google: {doc}")
        elif is_image(doc):
            # https://ai.google.dev/gemini-api/docs/image-understanding
            img_type, img_b64 = get_type_and_b64(
                doc,
                allowed=[
                    "image/png",
                    "image/jpeg",
                    "image/webp",
                    "image/heic",
                    "image/heif",
                ],
            )
            formatted_docs.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": img_type,
                                "data": img_b64,
                            }
                        }
                    ],
                }
            )
        else:
            raise ValueError(f"Unsupported document type: {type(doc)}")

    return formatted_docs


def _prepare_tool_schema(
    func_schemas: List[Union[dict, ServerTool]],
) -> List["types.ToolDict"]:
    """Convert tool definitions to Google Tool schema"""

    # if not isinstance(tools[0], types.Tool):
    #     tools = [types.Tool(function_declarations=[tool]) for tool in tools]
    google_tools: list["types.ToolDict"] = []
    for sch in func_schemas:
        if isinstance(sch, ServerTool):
            # kwargs can be specialized tool parameters
            extra_params: Union[
                "types.GoogleSearchDict", "types.ToolCodeExecutionDict"
            ] = sch.kwargs or {}
            if sch.server_tool_type == "web_search":
                google_tools.append({"google_search": extra_params})
            elif sch.server_tool_type == "code_interpreter":
                google_tools.append({"code_execution": extra_params})
            elif sch.server_tool_type == "mcp":
                # https://ai.google.dev/gemini-api/docs/interactions#remote-mcp-model-context-protocol
                # No type hint (?) for mcp tool
                obj = {
                    "type": "mcp_server",
                    "name": sch.server_label,
                    # "server_description": sch.server_description,
                    "url": sch.server_url,
                    # "require_approval": sch.require_approval,
                }
                google_tools.append(obj)
            else:
                raise ValueError(
                    f"Unsupported ServerTool type for Google Gemini: {sch.server_tool_type}"
                )
            continue
        from google.genai import types

        if isinstance(sch, types.Tool):
            google_tools.append(sch)
            continue
        if "type" in sch:
            # remove type field (used by openai)
            sch = sch.copy()
            sch.pop("type")

        # function_declarations = [types.FunctionDeclaration(**tool)]
        # google_tool = types.Tool(function_declarations=[sch])
        function_tool: "types.ToolDict" = {"function_declarations": [sch]}
        google_tools.append(function_tool)
    return google_tools


def _prepare_google_config(params: CommonQueryParameters, **kwargs):
    """Prepare config and contents for Google API calls"""
    instructions = params["instructions"]
    llm = params["llm"]
    structured_output = params.get("structured_output")
    tools = params.get("tools")

    contents = _fix_docs_for_google(params["strict_documents"])

    config: "types.GenerateContentConfigDict" = kwargs.copy()
    if instructions:
        config["system_instruction"] = instructions

    if structured_output is not None:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = structured_output

    if tools:
        config["tools"] = _prepare_tool_schema(tools)

    model_name = llm.model_name if llm else "gemini-2.5-flash"

    return model_name, contents, config


def _extract_text_from_gemini_model(resp: "BaseModel"):
    """Extract text content from Gemini API response.
    See google.genai.types.GenerateContentResponse._get_text
    Also suppresses a warning
    """
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
    # part.text == '' is different from part.text is None
    return text if any_text_part_text else None


def _extract_text_from_gemini_dict(resp: dict):
    """Extract text content from Gemini API response.
    See google.genai.types.GenerateContentResponse._get_text
    Also suppresses a warning
    """
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
    # part.text == '' is different from part.text is None
    return text if any_text_part_text else None


class GoogleProvider(BaseProvider):
    provider_type: str = "google"

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
        self, raw_response: Union["BaseModel", dict], provider_type: str = None
    ) -> ParsedResponse:
        """Parse Gemini API response into common format"""
        if isinstance(raw_response, dict):
            resp_id = raw_response.pop("response_id", None) or raw_response.pop(
                "responseId", None
            )

            # Extract text from Gemini response
            # from google.genai.types.GenerateContentResponse import _get_text
            text_content = _extract_text_from_gemini_dict(raw_response)

            tools = []
            for part in (
                raw_response.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            ):
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
                            call_id=func_call.get("id"),
                        )
                    )

            if text_content is None:
                text_content = ""  # I guess this can happen
            return ParsedResponse(
                text=text_content,
                response_id=resp_id,
                metadata=raw_response,
                function_calls=tools,
            )
        elif is_pydantic_model(raw_response):
            # Pydantic model (e.g., google.genai.types.GenerateContentResponse)

            response: "types.GenerateContentResponse" = raw_response
            # Suppress warning
            # text = response.text
            text = _extract_text_from_gemini_model(response)
            response_id = response.response_id
            obj = response.model_dump(mode="json")
            obj.pop("response_id", None)

            tools = []

            if not response.candidates:
                # Could be an error like rate limit
                pass

            for part in response.candidates[0].content.parts or []:
                if part.function_call:
                    func_call: "types.FunctionCall" = part.function_call
                    tools.append(
                        FunctionCall(
                            name=func_call.name,
                            arguments=func_call.args,
                            call_id=func_call.id,
                        )
                    )
            if text is None:
                text = ""  # I guess this can happen
            return ParsedResponse(
                text=text, response_id=response_id, metadata=obj, function_calls=tools
            )
        else:
            raise ValueError(f"Unsupported response type: {type(raw_response)}")


class SyncGoogleProvider(SyncProvider, GoogleProvider):
    def __init__(self, client: "genai.Client"):
        self.client = client

    def prepare_sync_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a synchronous callable for Gemini API"""
        model_name, contents, config = _prepare_google_config(params, **kwargs)

        # from google.genai.errors import ClientError

        return self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )


class ConcurrentGoogleProvider(ConcurrentProvider, GoogleProvider):
    def __init__(self, client: "genai.Client"):
        self.client = client

    def prepare_concurrent_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a concurrent coroutine for Gemini API"""
        model_name, contents, config = _prepare_google_config(params, **kwargs)

        coro = self.client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        return coro


def _camel_case_items(items: dict) -> dict:
    """Convert keys in a dictionary to camelCase."""
    if isinstance(items, list):
        return [_camel_case_items(item) for item in items]
    if not isinstance(items, dict):
        return items

    result = {}
    for k, v in items.items():
        if k == "properties":
            # each key in properties is custom, and should be untouched
            result[k] = {
                prop_k: _camel_case_items(prop_v) for prop_k, prop_v in v.items()
            }

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


class BatchGoogleProvider(BatchProvider, GoogleProvider):
    def __init__(self, client: "genai.Client"):
        self.client = client

    def prepare_batch_call(
        self,
        params: CommonQueryParameters,
        custom_id: str,
        **kwargs,
    ):
        """Convert CommonQueryParameters to Gemini batch request format"""
        model_name, fixed_documents, config = _prepare_google_config(params, **kwargs)

        # some differences: generationConfig
        # convert pydantic schema -> json
        gen_config = {}
        if "response_schema" in config:
            sch = config.pop("response_schema")
            if not isinstance(sch, dict):
                if getattr(sch, "model_json_schema", None):
                    sch = sch.model_json_schema()
            gen_config["responseJsonSchema"] = sch
        if "response_mime_type" in config:
            gen_config["responseMimeType"] = config.pop("response_mime_type")
        if gen_config:
            config["generationConfig"] = gen_config

        # For some reason, batch API uses slightly different
        # enums must be capitalized
        for tool in config.get("tools") or []:
            for decl in tool.get("function_declarations") or []:
                if "parameters" in decl:
                    _capitalize_function_decl(decl["parameters"])

        # Gemini batch request format
        request = {
            "key": custom_id,
            "request": {
                # https://ai.google.dev/api/batch-api#GenerateContentRequest
                # "contents": fixed_documents,
                # **config,
                "contents": _camel_case_items(fixed_documents),
                **_camel_case_items(config),
                # "generationConfig": gen_config,
            },
        }

        return request

    def _decode_gemini_batch_result(
        self, result: dict, custom_id: str
    ) -> ParsedResponse:
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

    def _decode_gemini_batch_error(
        self, result: dict, custom_id: str
    ) -> ParsedResponse:
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
                        parsed_errors.append(
                            self._decode_gemini_batch_error(line_data, custom_id)
                        )
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
                    file_content = self.client.files.download(
                        file=batch_job.dest.file_name
                    )
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
            error_msg = getattr(
                batch_job, "error", f"Job state: {batch_job.state.name}"
            )
            results.append(
                BatchResult(
                    status="error",
                    raw_output=str(error_msg),
                    parsed_responses=None,
                )
            )

        return results
