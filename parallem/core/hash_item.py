import hashlib
from io import BytesIO
import json
from typing import Union

from parallem.types import DocumentType, FileInput, FunctionCallOutput, FunctionCallRequest, LLMDocument, LLMResponse, MCPOutput, MultipartDocument, to_serial_id
from parallem.utils.image import is_image


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_str(text: str) -> str:
    return _hash_bytes(text.encode("utf-8", errors="replace"))


def _hash_part(part: LLMDocument) -> tuple[str, DocumentType]:
    """
    :return (item_hash, item_type)
    """

    if isinstance(part, str):
        return _hash_str(f"text\x00\x00{part}"), "text"

    elif is_image(part):
        buffered = BytesIO()
        part.save(buffered, format=part.format or "PNG")
        img_bytes = buffered.getvalue()
        item_hash = _hash_bytes(b"image\x00" + img_bytes)
        return _hash_bytes(b"image\x00" + img_bytes), "image"


    if part is None or isinstance(part, (int, float, bool, dict, list)):
        json_text = json.dumps(part, separators=(",", ":"))
        return _hash_str(f"json\x00{json_text}"), "json"


    if isinstance(part, FileInput):
        raw = (part.file_content or b"") + (part.file_url or "").encode()
        item_hash = _hash_bytes(b"input_file\x00" + raw + (part.filename or "").encode())
        return item_hash, "input_file"

    return None, None
        

def hash_item(
    msg: Union[LLMDocument, LLMResponse],
) -> tuple[str, DocumentType]:
    if isinstance(msg, MultipartDocument):
        part_hashes = []
        if len(msg.parts) == 1:
            # if there's only one part,
            # make sure that MultipartDocument has identical hash to
            # that single part
            if msg.role == "user":
                return _hash_part(msg.parts[0])
            else:
                return hash_item((msg.role, msg.parts[0]))
        for part in msg.parts:
            part_hashes.append(_hash_part(part))
        hash_concat = "\x00".join(part_hashes)
        return _hash_str(f"multipart\x00{hash_concat}"), "multipart"
    if isinstance(part, tuple) and len(part) == 2:
        role, text = part
        return _hash_str(f"text\x00{role}\x00{text}"), "text"
    if isinstance(msg, (FunctionCallOutput, MCPOutput)):
        # best effort to obtain hash (irreversible): take str(x)
        fc_content = str(msg.content)
        return _hash_str(
            f"function_call_output\x00{msg.name}\x00{fc_content}\x00{msg.fcall_id}"
        ), "function_call_output"

    if isinstance(msg, FunctionCallRequest):
        to_hash = "function_call\x00" + (msg.text_content or "")
        for call in msg.calls:
            to_hash += _hash_str(f"{call.name}\x00{call.arg_str}\x00{call.fcall_id}")
        return _hash_str(to_hash), "function_call"
    if isinstance(msg, LLMResponse):
        call_id = to_serial_id(msg.call_id) if msg.call_id is not None else None
        return _hash_str(f"llm_response\x00{call_id}"), "llm_response"
