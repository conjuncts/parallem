import hashlib
from typing import List, Optional
from io import BytesIO
from PIL import Image

from pipelinellm.types import LLMDocument, FunctionCallRequest, FunctionCallOutput


def _updateh(hasher, val: Optional[str]):
    if val is not None:
        hasher.update(val.encode("utf-8"))


def compute_hash(instructions: Optional[str], documents: List[LLMDocument]) -> str:
    """
    Compute a hash for the given instructions and documents.

    :param instructions: The instructions to hash.
    :param documents: The documents to hash.
    :returns: A SHA-256 hash representing the combined content, in hexadecimal format.
    """
    hasher = hashlib.sha256()
    if instructions:
        hasher.update(instructions.encode("utf-8"))
    for doc in documents:
        if isinstance(doc, str):
            hasher.update(doc.encode("utf-8"))
        elif isinstance(doc, Image.Image):
            with BytesIO() as img_buffer:
                doc.save(img_buffer, format="PNG")
                hasher.update(img_buffer.getvalue())
        elif isinstance(doc, FunctionCallRequest):
            hasher.update(b"function_call")
            _updateh(hasher, doc.text_content)
            for call in doc.calls:
                # hasher.update(str(call).encode("utf-8"))
                _updateh(hasher, call.name)
                _updateh(hasher, call.arg_str)
                _updateh(hasher, call.call_id)
        elif isinstance(doc, FunctionCallOutput):
            hasher.update(b"function_call_output")
            _updateh(hasher, doc.name)
            _updateh(hasher, str(doc.content))
            _updateh(hasher, doc.call_id)
        elif isinstance(doc, tuple) and len(doc) == 2:
            # Handle Tuple[Literal["user", "assistant", "system", "developer"], str]
            role, content = doc
            hasher.update(role.encode("utf-8"))
            if isinstance(content, str):
                hasher.update(content.encode("utf-8"))
            else:
                for item in content:
                    hasher.update(str(item).encode("utf-8"))
        else:
            raise ValueError(f"Unsupported document type: {type(doc)}")

    return hasher.hexdigest()
