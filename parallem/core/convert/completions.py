import base64
from io import BytesIO
from typing import TYPE_CHECKING, Literal, TypedDict, Union

from parallem.core.exception import ProviderCompatibilityError
from parallem.types import FileInput, FunctionCall, FunctionCallOutput, FunctionCallRequest, LLMDocument, MCPOutput, MultipartDocument
from parallem.utils.image import get_type_and_b64, is_image

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_content_part_param import ChatCompletionContentPartParam
    from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
    from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam
    from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
    from openai.types.chat.chat_completion_content_part_param import FileFile

class ChatCompletionPartConversionError(TypedDict):
    """
    Describes documents that cannot be converted to a ChatCompletionContentPartParam,
    describing a superset of the ChatCompletions API that is not officially supported.
    """
    type: Literal["unsupported"]

    reason: Literal[
        "file_url_not_supported",
        "cannot_convert_to_part",
    ]

class ChatCompletionConversionError(TypedDict):
    role: Literal["unsupported"]
    reason: Literal[
        "function_call_output_not_text_only",
    ]



def to_chat_completion_part(
    doc: LLMDocument,
) -> Union["ChatCompletionContentPartParam", "ChatCompletionPartConversionError"]:
    if isinstance(doc, str):
        return {"type": "text", "text": doc}
    elif is_image(doc):
        img_type, img_b64 = get_type_and_b64(doc)
        return {"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{img_b64}"}}
    elif isinstance(doc, tuple) and len(doc) == 2:
        return to_chat_completion_part(doc[1])
    elif isinstance(doc, FileInput):
        file_part: "FileFile" = {
            "filename": doc.filename,
        }
        if doc.file_url:
            # https://developers.openai.com/api/docs/guides/file-inputs?api-mode=chat
            return {
                "type": "unsupported",
                "reason": "file_url_not_supported",
            }
        if doc.file_content:
            b64 = base64.b64encode(doc.file_content).decode("utf-8")
            file_part["file_data"] = f"data:{doc.mime_type};base64,{b64}"
        return file_part
    else:
        # other types cannot be converted to Part, but instead should be converted to a full message
        return {
            "type": "unsupported",
            "reason": "cannot_convert_to_part",
        }

def to_chat_completion(
    doc: LLMDocument,
) -> Union[
    "ChatCompletionAssistantMessageParam",
    "ChatCompletionToolMessageParam",
    "ChatCompletionConversionError",
]:
    if isinstance(doc, str):
        return {
            "role": "user",
            "content": doc,
        }
    elif is_image(doc):
        return {
            "role": "user",
            "content": [to_chat_completion_part(doc)],
        }
    elif isinstance(doc, FileInput):
        return {
            "role": "user",
            "content": [to_chat_completion_part(doc)],
        }
    elif isinstance(doc, MultipartDocument):
        content_parts = []
        for part in doc.parts:
            converted = to_chat_completion_part(part)
            if isinstance(converted, dict) and converted.get("type") == "unsupported":
                return {
                    "role": "unsupported",
                    "reason": "cannot_convert_to_part",
                }
            content_parts.append(converted)
        return {
            "role": "user",
            "content": content_parts,
        }
    elif isinstance(doc, FunctionCallRequest):
        msg: "ChatCompletionAssistantMessageParam" = {
            "role": "assistant",
            "content": doc.text_content if doc.text_content else None,
            "tool_calls": [
                {
                    "id": call.fcall_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arg_str,
                    },
                }
                for call in doc.calls
            ],
        }
        return msg
    elif isinstance(doc, (FunctionCallOutput, MCPOutput)):
        # ChatCompletions requires that contents be text only
        if isinstance(doc.content, str) or (
            isinstance(doc.content, list) and (
                all(isinstance(c, str) for c in doc.content)
                or all(isinstance(c, dict) and c.get("type") == "text" for c in doc.content)
            )
        ):
            msg: "ChatCompletionToolMessageParam" = {
                "role": "tool",
                "tool_call_id": doc.fcall_id,
                "content": doc.content,
            }
            return msg
        else:
            return {
                "role": "unsupported",
                "reason": "function_call_output_not_text_only",
            }

def from_chat_completion(
    msg: Union[
        "ChatCompletionUserMessageParam",
        "ChatCompletionAssistantMessageParam",
        "ChatCompletionToolMessageParam",
    ]
) -> LLMDocument:
    def _decode_data_url(data_url: str) -> tuple[str, bytes]:
        if not data_url.startswith("data:") or ";base64," not in data_url:
            raise ProviderCompatibilityError(f"Unsupported data URL format: {data_url[:40]}...")
        header, b64 = data_url.split(";base64,", 1)
        mime_type = header.removeprefix("data:")
        return mime_type, base64.b64decode(b64)

    def _from_content_parts(parts: list) -> LLMDocument:
        converted_parts = []
        for part in parts:
            if isinstance(part, str):
                converted_parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    converted_parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url", {}).get("url")
                    if not isinstance(image_url, str):
                        raise ProviderCompatibilityError("image_url part is missing a URL")
                    _mime_type, raw = _decode_data_url(image_url)
                    from PIL import Image
                    converted_parts.append(Image.open(BytesIO(raw)))
                elif part.get("type") == "file" or "file" in part or "filename" in part:
                    file_part = part.get("file") if part.get("type") == "file" else part
                    if isinstance(file_part, dict) and isinstance(file_part.get("filename"), str):
                        filename = file_part["filename"]
                        file_data = file_part.get("file_data")
                        file_url = file_part.get("file_url")

                        if isinstance(file_data, str):
                            mime_type, raw = _decode_data_url(file_data)
                            converted_parts.append(FileInput(
                                mime_type=mime_type,
                                filename=filename,
                                file_content=raw,
                            ))
                        elif isinstance(file_url, str):
                            converted_parts.append(FileInput(
                                mime_type="application/octet-stream",
                                filename=filename,
                                file_url=file_url,
                            ))
                        else:
                            raise ProviderCompatibilityError(f"Unsupported file part: {part}")
                    else:
                        raise ProviderCompatibilityError(f"Unsupported file part structure: {part}")
                else:
                    raise ProviderCompatibilityError(f"Unsupported user content part: {part}")
            else:
                raise ProviderCompatibilityError(f"Unsupported user content part type: {type(part)}")

        if len(converted_parts) == 1:
            return converted_parts[0]
        return MultipartDocument(parts=converted_parts)

    if msg["role"] == "user":
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return _from_content_parts(content)
        raise ProviderCompatibilityError(f"Unsupported user message content: {content}")

    if msg["role"] == "assistant":
        calls = []
        if "tool_calls" in msg and msg["tool_calls"]:
            for call in msg["tool_calls"]:
                calls.append(
                    FunctionCall(
                        fcall_id=call["id"],
                        name=call["function"]["name"],
                        arguments=call["function"]["arguments"],
                    )
                )
        return FunctionCallRequest(
            text_content=msg.get("content") or "",
            calls=calls,
            call_id=None,
        )
    if msg["role"] == "tool":
        return FunctionCallOutput(
            content=msg.get("content"),
            fcall_id=msg["tool_call_id"],
            name="",
        )

    raise ProviderCompatibilityError(f"Unsupported chat completion role: {msg['role']}")