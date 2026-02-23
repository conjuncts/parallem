from typing import TYPE_CHECKING, List, Optional, Union
from pydantic import BaseModel
from parallellm.provider.base import ConcurrentProvider, BaseProvider, SyncProvider
from parallellm.types import (
    ParsedResponse,
    CommonQueryParameters,
    FunctionCallRequest,
    FunctionCallOutput,
    FunctionCall,
    LLMDocument,
    LLMIdentity,
    ServerTool,
)
from parallellm.utils.image import (
    _get_image_type,
    _image_to_b64,
    get_type_and_b64,
    is_image,
)

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import Message


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
                        "content": doc.content,
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
    tools = params.get("tools")

    if tools:
        tools = _prepare_tool_schema(tools)

    messages = _fix_docs_for_anthropic(params["strict_documents"])

    config = kwargs.copy()
    if instructions:
        config["system"] = instructions

    model_name = llm.model_name

    if tools is not None and len(tools) > 0:
        config["tools"] = tools
    return model_name, messages, config


def _prepare_tool_schema(func_schemas: List[Union[dict, ServerTool]]) -> List[dict]:
    """Convert tool definitions to Anthropic tool schema"""

    anthropic_tools = []
    for sch in func_schemas:
        if isinstance(sch, ServerTool):
            if sch.server_tool_type == "web_search":
                anthropic_tools.append(
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 5,
                        **sch.kwargs,
                    }
                )
            elif sch.server_tool_type == "code_interpreter":
                # TODO: support this (it is in beta)
                raise NotImplementedError
            elif sch.server_tool_type == "mcp":
                # TODO: support this (it is in beta)
                # https://platform.claude.com/docs/en/agents-and-tools/mcp-connector
                raise NotImplementedError
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

    return anthropic_tools


class AnthropicProvider(BaseProvider):
    provider_type: str = "anthropic"

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("claude-3-haiku-20240307", provider=self.provider_type)

    def parse_response(self, raw_response: Union[BaseModel, dict]) -> ParsedResponse:
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

        return self.client.messages.create(
            model=model_name,
            max_tokens=config.pop("max_tokens", 4096),
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

        coro = self.client.messages.create(
            model=model_name,
            max_tokens=config.pop("max_tokens", 1024),
            messages=messages,
            **config,
        )

        return coro
