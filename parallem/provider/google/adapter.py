import base64

from pydantic import BaseModel

from parallem.provider.base import BaseAdapter
from parallem.types import (
    CommonQueryParameters,
    FileInput,
    FunctionCall,
    FunctionCallOutput,
    FunctionCallRequest,
    LLMDocument,
    MCPOutput,
    MultipartDocument,
    ParsedResponse,
    ServerTool,
)

from collections.abc import Mapping
from typing import TYPE_CHECKING, List, Literal, Union, overload

from parallem.utils._quick_pydantic import is_pydantic_model
from parallem.utils.image import get_type_and_b64, is_image, is_image_url
from parallem.utils.manip import maybe_snake_to_camel


if TYPE_CHECKING:
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
        # https://ai.google.dev/gemini-api/docs/file-input-methods
        # Seems to be a discrepency between SDK and documentation.
        # for example, documentation recommends "type": "document"
        if part.file_content:
            return {
                "inline_data": {
                    "mime_type": part.mime_type,
                    "data": base64.b64encode(part.file_content).decode("utf-8"),
                    # display_name not supported in Gemini API
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
        # If it's already a proper content dict, keep it
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
    """
    Ensure documents are in the correct format for Gemini API.

    Note that the API allows a mixture of content and parts (types.ContentListUnionDict)
    :param documents: List of LLMDocument (str, tuple, dict, FileInput, FunctionCallRequest, FunctionCallOutput)
    :param strict: If True, always return types.ContentDict. 
        If False, allow returning types.PartDict or str, to more faithfully match
        the Gemini Python SDK.
    """

    # For Gemini, we can pass strings directly or convert to proper format
    # The SDK will handle the conversion automatically
    # if len(documents) == 1 and isinstance(documents[0], str):
    # return documents[0]  # Single string can be passed directly

    # For multiple documents or mixed types, return as list
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
            # https://ai.google.dev/gemini-api/docs/function-calling?example=meeting
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
            # types.Part()
            # types.FunctionResponse()
            # types.Content()
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
                # parts.append(types.Part(text=types.Text(content=doc.text_content)))
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
            # If it's already a proper content dict, keep it
            # needs to be Union[types.ContentDict, types.PartUnionDict]
            # (types str, PIL.Image)
            # = [types.ContentDict, types.FileDict, types.PartDict]
            # doc: Union["types.PartDict", "types.FileDict", "types.ContentDict"] = doc  # type hint for clarity
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

    # if not isinstance(tools[0], types.Tool):
    #     tools = [types.Tool(function_declarations=[tool]) for tool in tools]
    google_tools: list["types.ToolDict"] = []
    for sch in func_schemas:
        if isinstance(sch, ServerTool):
            # kwargs can be specialized tool parameters
            extra_params: Union["types.GoogleSearchDict", "types.ToolCodeExecutionDict"] = (
                sch.kwargs or {}
            )
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
        elif isinstance(sch, Mapping):
            if "type" in sch:
                # remove type field (used by openai)
                sch = sch.copy()
                sch.pop("type")
            # function_declarations = [types.FunctionDeclaration(**tool)]
            # google_tool = types.Tool(function_declarations=[sch])
            function_tool: "types.ToolDict" = {"function_declarations": [sch]}
            google_tools.append(function_tool)
        else:
            google_tools.append(sch)
    return google_tools


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

        # some differences between python SDK and JSON.

        # In essence, The python SDK expects GenerateContentConfigDict
        # (which is in snake_case)
        # But the Batch API expects the JSON
        # (which is in camelCase)
        
        # JSON version also expects the following to be bubbled up:
        # model, contents, tools, toolConfig, safetySettings,
        # systemInstruction, cachedContent, serviceTier, store

        # The Python SDK flattens what should be 2 nested
        # JSON configs, which must be reversed.
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

        # this way, if the user provides config in kwargs under
        # generation_config, it gets observed too
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

        # For some reason, batch API uses slightly different
        # enums must be capitalized
        for tool in big_config.get("tools") or []:
            for decl in tool.get("functionDeclarations") or []:
                if "parameters" in decl:
                    _capitalize_function_decl(decl["parameters"])

        body = {
            # https://ai.google.dev/api/batch-api#GenerateContentRequest
            "contents": _camel_case_items(fixed_documents),
            **big_config,
        }
        return body

    def convert_response(self, raw_response: Union["BaseModel", dict]) -> ParsedResponse:
        """Parse Gemini API response into common format"""
        if isinstance(raw_response, dict):
            resp_id = raw_response.get("response_id") or raw_response.get("responseId")

            # Extract text from Gemini response
            # from google.genai.types.GenerateContentResponse import _get_text
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
                            fcall_id=func_call.id,
                        )
                    )
            if text is None:
                text = ""  # I guess this can happen
            return ParsedResponse(
                text=text, response_id=response_id, metadata=obj, function_calls=tools
            )
        else:
            raise ValueError(f"Unsupported response type: {type(raw_response)}")
