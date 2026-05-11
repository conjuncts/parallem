import json
import importlib.metadata as importlib_metadata
import copy
from pathlib import Path
from typing import TYPE_CHECKING, List, Union
from pydantic import BaseModel
from parallem.core.exception import ProviderCompatibilityError
from parallem.provider.base import (
    BatchProvider,
    ConcurrentProvider,
    BaseProvider,
    SyncProvider,
)
from parallem.types import (
    BatchResult,
    ParsedResponse,
    ParsedError,
    CommonQueryParameters,
    FunctionCallRequest,
    FunctionCallOutput,
    FunctionCall,
    LLMDocument,
    LLMIdentity,
    ServerTool,
)
from parallem.utils._batch_helper import _split_batch_response
from parallem.utils.image import (
    get_type_and_b64,
    is_image,
)
from parallem.provider.openai.openai_tools import (
    _ensure_strict_json_schema,
    to_strict_json_schema,
)

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import Message
    from parallem.tools.mcp import MCPTool


_ANTHROPIC_STRUCTURED_OUTPUT_MIN_VERSION = "0.77.0"


def _numeric_version_triplet(version_text: str) -> tuple[int, int, int]:
    numeric_parts = []
    current = ""

    for char in version_text:
        if char.isdigit():
            current += char
            continue
        if current:
            numeric_parts.append(int(current))
            current = ""
            if len(numeric_parts) >= 3:
                break

    if current and len(numeric_parts) < 3:
        numeric_parts.append(int(current))

    while len(numeric_parts) < 3:
        numeric_parts.append(0)

    return tuple(numeric_parts[:3])


def _enforce_anthropic_min_version_for_structured_output() -> None:
    """Validate installed anthropic package supports output_config.format."""

    try:
        installed_version_text = importlib_metadata.version("anthropic")
    except importlib_metadata.PackageNotFoundError as exc:
        raise ImportError(
            "Structured output with Anthropic requires the 'anthropic' package to be installed."
        ) from exc

    installed_version = _numeric_version_triplet(installed_version_text)
    minimum_required = _numeric_version_triplet(
        _ANTHROPIC_STRUCTURED_OUTPUT_MIN_VERSION
    )
    if installed_version < minimum_required:
        raise ProviderCompatibilityError(
            "Structured output with Anthropic requires anthropic>="
            f"{_ANTHROPIC_STRUCTURED_OUTPUT_MIN_VERSION}, found {installed_version_text}. "
            "Please upgrade the anthropic package."
        )


def _ensure_betas(config: dict, betas_to_add: Union[str, List[str]] | None) -> None:
    """Ensure required beta feature flags are present in the outgoing config.

    This helper is defensive and accepts either a single beta string or a list
    of betas. It normalizes the `config["betas"]` value into a list and
    appends any missing entries while preserving any pre-existing values.

    :param config: Mutable request config dict to update.
    :param betas_to_add: A string or list of strings to add to `config['betas']`.
    :return: None (modifies `config` in-place).
    """
    if not betas_to_add:
        return

    if isinstance(betas_to_add, str):
        required = [betas_to_add]
    else:
        required = list(betas_to_add)

    existing = config.get("betas")
    if existing is None:
        config["betas"] = []
    elif isinstance(existing, list):
        # use as-is
        pass
    else:
        config["betas"] = [existing]

    for b in required:
        if b not in config["betas"]:
            config["betas"].append(b)


def _fix_docs_for_anthropic(
    documents: List[LLMDocument],
) -> List[dict]:
    """Ensure documents are in the correct format for Anthropic API"""

    formatted_docs = []
    for doc in documents:
        if isinstance(doc, str):
            msg = {
                "role": "user",
                "content": doc,
            }
            formatted_docs.append(msg)
            continue
        elif isinstance(doc, FunctionCallRequest):
            msg_contents = []
            if doc.text_content:
                msg_contents.append(
                    {
                        "type": "text",
                        "text": doc.text_content,
                    }
                )
            msg_contents += [
                {
                    "type": "tool_use",
                    "name": call.name,
                    "input": call.args,
                    "id": call.call_id,
                }
                for call in doc.calls
            ]
            msg = {"role": "assistant", "content": msg_contents}
            formatted_docs.append(msg)
            continue
        elif isinstance(doc, FunctionCallOutput):
            msg = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": doc.call_id,
                        "content": str(doc.content),
                    }
                ],
            }
            formatted_docs.append(msg)
            continue
        elif isinstance(doc, tuple) and len(doc) == 2:
            # from anthropic.types.message_param import MessageParam
            # For anthropic, only valid roles are ["user", "assistant"]

            role, content = doc
            if role in {"system", "developer"}:
                # Anthropic does not have system/developer roles, map to user
                msg = {
                    "role": "user",
                    "content": content,
                }
            elif role in {"user", "assistant"}:
                msg = {
                    "role": role,
                    "content": content,
                }
            else:
                raise ValueError(f"Unsupported role in document tuple: {role}")
            formatted_docs.append(msg)
            continue
        elif isinstance(doc, dict):
            # If it's already a proper message dict, keep it
            if "role" in doc and "content" in doc:
                formatted_docs.append(doc)
                continue
        elif is_image(doc):
            # https://platform.claude.com/docs/en/build-with-claude/vision
            img_type, img_b64 = get_type_and_b64(
                doc, allowed=["image/jpeg", "image/png", "image/gif", "image/webp"]
            )
            formatted_docs.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": img_type,
                                "data": img_b64,
                            },
                        },
                    ],
                }
            )
            continue
        raise ValueError(f"Unsupported document type: {type(doc)}")

    # TODO: roll up consecutive messages from the same role, especially for assistant role
    # and for tool calls
    return formatted_docs


def _prepare_anthropic_config(params: CommonQueryParameters, **kwargs) -> tuple:
    """Prepare config and messages for Anthropic API calls"""
    instructions = params["instructions"]
    llm = params["llm"]
    structured_output = params.get("structured_output")
    tools = params.get("tools")

    mcp_servers: list[dict] = []
    if tools:
        tools, mcp_servers = _prepare_tool_schema(tools)

    messages = _fix_docs_for_anthropic(params["strict_documents"])

    config = kwargs.copy()
    if instructions:
        config["system"] = instructions

    if structured_output is not None:
        if (config.get("output_config") or {}).get("format") is not None:
            raise AssertionError(
                "Cannot supply both structured_output and output_config.format"
            )

        config["output_config"] = config.get("output_config", {})
        config["output_config"]["format"] = _prepare_anthropic_output_format(
            structured_output
        )

    model_name = llm.model_name

    if mcp_servers:
        config["mcp_servers"] = mcp_servers
        _ensure_betas(config, "mcp-client-2025-11-20")

    if tools is not None and len(tools) > 0:
        config["tools"] = tools
    return model_name, messages, config


def _prepare_anthropic_output_format(structured_output: object) -> dict:
    """Prepare Anthropic output_config.format payload from structured_output input."""

    def _strict_schema(schema: dict) -> dict:
        schema_copy = copy.deepcopy(schema)
        return _ensure_strict_json_schema(schema_copy, path=(), root=schema_copy)

    def _strict_format(format_dict: dict) -> dict:
        strict_format = copy.deepcopy(format_dict)
        schema = strict_format.get("schema")
        if isinstance(schema, dict):
            strict_format["schema"] = _strict_schema(schema)
        return strict_format

    if isinstance(structured_output, dict):
        if (
            structured_output.get("type") == "json_schema"
            and "schema" in structured_output
        ):
            return _strict_format(structured_output)
        if "format" in structured_output and isinstance(
            structured_output["format"], dict
        ):
            return _strict_format(structured_output["format"])
        return {
            "type": "json_schema",
            "schema": _strict_schema(structured_output),
        }

    model_json_schema = getattr(structured_output, "model_json_schema", None)
    if callable(model_json_schema):
        return {
            "type": "json_schema",
            "schema": to_strict_json_schema(structured_output),
        }

    raise ValueError(
        "Unsupported structured_output for Anthropic. Expected dict JSON schema or a Pydantic model/class with model_json_schema()."
    )


def _prepare_tool_schema(
    func_schemas: List[Union[dict, ServerTool]],
) -> tuple[list[dict], list[dict]]:
    """Convert tool definitions to Anthropic tool schema"""

    anthropic_tools: list[dict] = []
    mcp_servers: list[dict] = []
    for sch in func_schemas:
        if isinstance(sch, ServerTool):
            if sch.server_tool_type == "web_search":
                anthropic_tools.append(
                    {
                        "type": "web_search_20260209",
                        "name": "web_search",
                        "max_uses": 5,
                        **sch.kwargs,
                    }
                )
            elif sch.server_tool_type == "code_interpreter":
                # TODO: support this (it is in beta)
                raise NotImplementedError
            elif sch.server_tool_type == "mcp":
                sch: "MCPTool"
                # https://platform.claude.com/docs/en/agents-and-tools/mcp-connector
                mcp_server = {
                    "type": "url",
                    "url": sch.server_url,
                    "name": sch.server_label,
                }
                if sch.authorization_token is not None:
                    mcp_server["authorization_token"] = sch.authorization_token
                mcp_servers.append(mcp_server)

                toolset = {
                    "type": "mcp_toolset",
                    "mcp_server_name": sch.server_label,
                }
                if sch.default_config is not None:
                    toolset["default_config"] = sch.default_config
                if sch.configs is not None:
                    toolset["configs"] = sch.configs
                if sch.cache_control is not None:
                    toolset["cache_control"] = sch.cache_control
                anthropic_tools.append(toolset)
            else:
                raise ValueError(
                    f"Unsupported ServerTool type for Anthropic: {sch.server_tool_type}"
                )
            continue

        sch2 = None
        if "type" in sch:
            # remove type field (used by openai)
            sch2 = sch.copy()
            sch2.pop("type")

        if "parameters" in sch:
            sch2 = sch2 or sch.copy()
            # rename parameters to input_schema
            sch2["input_schema"] = sch2.pop("parameters")

        if sch2 is not None:
            sch = sch2
        anthropic_tools.append(sch)

    return anthropic_tools, mcp_servers


class AnthropicProvider(BaseProvider):
    provider_type: str = "anthropic"

    def validate_request_compatibility(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> None:
        """Validate Anthropic request compatibility.

        :param params: Common query parameters for the request.
        :return: None.
        """
        structured_output = params.get("structured_output")
        if structured_output is not None:
            _enforce_anthropic_min_version_for_structured_output()

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity(
            "claude-haiku-4-5-20251001", provider_type=self.provider_type
        )

    def parse_response(
        self, raw_response: Union[BaseModel, dict], provider_type: str = None
    ) -> ParsedResponse:
        """Parse Anthropic API response into common format"""

        # https://docs.claude.com/en/docs/agents-and-tools/tool-use/implement-tool-use
        if isinstance(raw_response, BaseModel):
            # Pydantic model (e.g., anthropic.types.Message)
            response: Message = raw_response
            text_contents = []
            tool_calls = []
            for content_item in response.content:
                if content_item.type == "text":
                    text_contents.append(content_item.text)
                elif content_item.type == "tool_use":
                    tool_calls.append(
                        FunctionCall(
                            name=content_item.name,
                            arguments=content_item.input,
                            call_id=content_item.id,
                        )
                    )
            text_content = "".join(text_contents)

            resp_id = response.id
            obj = response.model_dump(mode="json")
            obj.pop("id", None)
            parsed_metadata = obj

        elif isinstance(raw_response, dict):
            # Dict response
            resp_id = raw_response.get("id", None)

            # Extract text and tool calls from content
            content = raw_response.get("content", [])
            text_contents = []
            tool_calls = []

            if isinstance(content, list):
                for content_item in content:
                    if not isinstance(content_item, dict):
                        text_contents.append(str(content_item))
                        continue

                    if content_item.get("type") == "text":
                        text_contents.append(content_item["text"])
                    elif content_item.get("type") == "tool_use":
                        tool_calls.append(
                            FunctionCall(
                                name=content_item.get("name"),
                                arguments=content_item.get("input"),
                                call_id=content_item.get("id"),
                            )
                        )
            else:
                text_contents.append(str(content))

            text_content = "".join(text_contents)

            # Create a copy for metadata to avoid mutating the original
            parsed_metadata = raw_response.copy()
            parsed_metadata.pop("id", None)
        else:
            raise ValueError(f"Unsupported response type: {type(raw_response)}")
        return ParsedResponse(
            text=text_content,
            response_id=resp_id,
            metadata=parsed_metadata,
            function_calls=tool_calls,
        )


class SyncAnthropicProvider(SyncProvider, AnthropicProvider):
    def __init__(self, client: "Anthropic"):
        self.client = client

    def prepare_sync_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a synchronous callable for Anthropic API"""
        model_name, messages, config = _prepare_anthropic_config(params, **kwargs)
        max_tokens = config.pop("max_tokens", 4096)
        # If betas are requested and the client exposes the beta namespace,
        # use the beta messages.create endpoint.
        if config.get("betas"):
            return self.client.beta.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=messages,
                **config,
            )

        return self.client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=messages,
            **config,
        )


class ConcurrentAnthropicProvider(ConcurrentProvider, AnthropicProvider):
    def __init__(self, client: "AsyncAnthropic"):
        self.client = client

    def prepare_concurrent_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a concurrent coroutine for Anthropic API"""
        model_name, messages, config = _prepare_anthropic_config(params, **kwargs)
        max_tokens = config.pop("max_tokens", 1024)

        if config.get("betas"):
            coro = self.client.beta.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=messages,
                **config,
            )
        else:
            coro = self.client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=messages,
                **config,
            )

        return coro


class BatchAnthropicProvider(BatchProvider, AnthropicProvider):
    def __init__(self, client: "Anthropic"):
        self.client = client

    def prepare_batch_call(
        self,
        params: CommonQueryParameters,
        custom_id: str,
        **kwargs,
    ):
        """Convert CommonQueryParameters to Anthropic Message Batch format."""
        model_name, messages, config = _prepare_anthropic_config(params, **kwargs)

        request_params = {
            "model": model_name,
            "max_tokens": config.pop("max_tokens", 4096),
            "messages": messages,
            **config,
        }

        return {
            "custom_id": custom_id,
            "params": request_params,
        }

    def get_batch_custom_ids(self, stuff: list[dict], provider_type: str) -> list[str]:
        custom_ids = []
        for item in stuff:
            custom_id = item.get("custom_id")
            if not custom_id:
                raise ValueError("Missing custom_id in batch item")
            custom_ids.append(custom_id)
        return custom_ids

    def submit_batch_to_provider(self, fpath: Path, llm: LLMIdentity) -> str:
        """Submit JSONL requests as an Anthropic Message Batch and return batch ID."""
        requests = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                requests.append(json.loads(line))

        batch = self.client.messages.batches.create(requests=requests)
        return batch.id

    def cancel_batch(self, batch_uuid: str, provider_type: str) -> None:
        """Cancel an Anthropic Message Batch."""
        self.client.messages.batches.cancel(batch_uuid)

    def _decode_anthropic_batch_success(self, line_data: dict) -> ParsedResponse:
        custom_id = line_data.get("custom_id")
        result = line_data.get("result") or {}
        message_obj = result.get("message")

        if not isinstance(message_obj, dict):
            raise ValueError(
                "Missing succeeded message payload in Anthropic batch line"
            )

        parsed = self.parse_response(message_obj)
        parsed.custom_id = custom_id
        return parsed

    def _decode_anthropic_batch_error(self, line_data: dict) -> ParsedResponse:
        custom_id = line_data.get("custom_id")
        result = line_data.get("result") or {}
        result_type = result.get("type") or "errored"

        error_obj = result.get("error")
        if not isinstance(error_obj, dict):
            error_obj = {"type": result_type}

        error_message = error_obj.get("message") or error_obj.get("type") or result_type

        status_code = error_obj.get("status_code")
        if isinstance(status_code, int):
            error_code = status_code
        elif isinstance(status_code, str) and status_code.isdigit():
            error_code = int(status_code)
        else:
            error_code = 1

        return ParsedResponse(
            text=str(error_message),
            response_id=None,
            custom_id=custom_id,
            metadata=error_obj,
            error_code=error_code,
        )

    def decode_batch_content(self, content: str) -> List[BatchResult]:
        """Decode Anthropic JSONL batch output into BatchResult objects."""
        parsed_responses = []
        parsed_errors = []
        not_ok_i = []

        for line_i, line in enumerate(content.strip().split("\n")):
            if not line:
                continue
            try:
                line_data = json.loads(line)
                result_type = (line_data.get("result") or {}).get("type")

                if result_type == "succeeded":
                    parsed_responses.append(
                        self._decode_anthropic_batch_success(line_data)
                    )
                else:
                    parsed_errors.append(self._decode_anthropic_batch_error(line_data))
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
        """Download Anthropic Message Batch results.

        :param batch_uuid: The UUID of the batch to download.
        :return: List of BatchResult objects containing the results and errors (if any).
            If nothing is ready yet, empty list is returned.
        """

        batch = self.client.messages.batches.retrieve(batch_uuid)
        if batch.processing_status != "ended":
            return []

        lines = []
        for item in self.client.messages.batches.results(batch_uuid):
            if isinstance(item, dict):
                lines.append(json.dumps(item))
            elif isinstance(item, str):
                lines.append(item)
            else:
                dump_json = getattr(item, "model_dump_json", None)
                if callable(dump_json):
                    lines.append(dump_json())
                else:
                    model_dump = getattr(item, "model_dump", None)
                    if callable(model_dump):
                        lines.append(json.dumps(model_dump(mode="json")))
                    else:
                        lines.append(json.dumps(item))

        content = "\n".join(lines)
        if not content:
            return []
        return self.decode_batch_content(content)
