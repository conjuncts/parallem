import copy
from typing import TYPE_CHECKING, List, Union
from anthropic.types.document_block_param import DocumentBlockParam
from anthropic.types.image_block_param import ImageBlockParam
from anthropic.types.text_block_param import TextBlockParam
from anthropic.types.tool_result_block_param import ToolResultBlockParam
from pydantic import BaseModel
from parallem.provider.base import (
    BaseParser,
)
from parallem.types import (
    MCPOutput,
    ParsedResponse,
    CommonQueryParameters,
    FunctionCallRequest,
    FunctionCallOutput,
    FunctionCall,
    LLMDocument,
    ServerTool,
)
from parallem.utils.image import (
    get_type_and_b64,
    is_image,
)

if TYPE_CHECKING:
    from anthropic.types import Message
    from anthropic.types.message_param import MessageParam
    from parallem.tools.mcp import MCPServerTool
    from anthropic.types.tool_result_block_param import ToolResultBlockParam
    from mcp.types import ContentBlock


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
            if isinstance(doc, MCPOutput):
                fc_content = [_fix_mcp_block(x) for x in doc.content]
            else:
                fc_content = doc.content
            tool_result_content: "ToolResultBlockParam" = {
                "type": "tool_result",
                "tool_use_id": doc.call_id,
                "content": fc_content,
            }
            msg = {
                "role": "user",
                "content": [tool_result_content],
            }
            formatted_docs.append(msg)
            continue
        elif isinstance(doc, tuple) and len(doc) == 2:
            # For anthropic, only valid roles are ["user", "assistant"]

            role, content = doc
            if role in {"system", "developer"}:
                # Anthropic does not have system/developer roles, map to user
                msg: "MessageParam" = {
                    "role": "user",
                    "content": content,
                }
            elif role in {"user", "assistant"}:
                msg: "MessageParam" = {
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


def _fix_mcp_block(
    content_block: "ContentBlock"
) -> "ToolResultBlockParam":
    # Overall type:
    if content_block.type == "text":
        output: "TextBlockParam" = {
            "type": "text",
            "text": content_block.text,
        }
        return output
    if content_block.type == "image":
        img_type = content_block.mimeType
        img_b64 = content_block.data
        # "ResponseInputImage"
        output: "ImageBlockParam" = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img_type,
                "data": img_b64,
            },
        }
        return output
    if content_block.type == "resource":
        # "ResponseInputFile"
        # TODO untested
        resource = content_block.resource
        if getattr(resource, "blob"):
            content = resource.blob
        else:
            content = resource.text
        output: "DocumentBlockParam" = {
            "type": "document",
            "source": content
        }
    return content_block.model_dump_json(exclude_none=True)


def _transform_schema(base_model) -> dict:
    import anthropic
    return anthropic.transform_schema(base_model)


def _prepare_anthropic_output_format(structured_output: object) -> dict:
    """Prepare Anthropic output_config.format payload from structured_output input."""

    def _strict_format(format_dict: dict) -> dict:
        strict_format = copy.deepcopy(format_dict)
        schema = strict_format.get("schema")
        if isinstance(schema, dict):
            strict_format["schema"] = _transform_schema(schema)
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
            "schema": _transform_schema(structured_output),
        }

    model_json_schema = getattr(structured_output, "model_json_schema", None)
    if callable(model_json_schema):
        return {
            "type": "json_schema",
            "schema": _transform_schema(structured_output),
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
                sch: "MCPServerTool"
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


class AnthropicParser(BaseParser):
    def fix_docs(
        self,
        documents: List[LLMDocument],
    ):
        return _fix_docs_for_anthropic(documents)

    def fix_tools(
        self,
        tools: List[Union[dict, ServerTool]],
    ) -> tuple[list[dict], list[dict]]:
        return _prepare_tool_schema(tools)

    def fix_config(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> tuple[str, List[dict], dict]:
        """Prepare config and messages for Anthropic API calls"""
        instructions = params["instructions"]
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = params.get("tools")

        mcp_servers: list[dict] = []
        if tools:
            tools, mcp_servers = self.fix_tools(tools)

        messages = self.fix_docs(params["strict_documents"])

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

    def prepare_request(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> dict:
        """Prepare full request payload for Anthropic API calls"""
        model_name, messages, config = self.fix_config(params, **kwargs)

        request_params = {
            "model": model_name,
            "max_tokens": config.pop("max_tokens", 4096),
            "messages": messages,
            **config,
        }
        return request_params

    def convert_response(
        self, raw_response: Union[BaseModel, dict], provider_type: str = None
    ) -> ParsedResponse:
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