import hashlib
from typing import List, Optional
from io import BytesIO
from PIL import Image
import warnings

from parallem.types import LLMDocument, FunctionCallRequest, FunctionCallOutput


def _updateh(hasher, val: Optional[str]):
    if val is not None:
        hasher.update(val.encode("utf-8"))


def compute_hash(
    instructions: Optional[str],
    documents: List[LLMDocument],
    *,
    salt: Optional[str] = None,
    _legacy_salt: Optional[List[str]] = None,
) -> str:
    """
    Compute a hash for the given instructions and documents.

    :param instructions: The instructions to hash.
    :param documents: The documents to hash.
    :param salt: An optional salt string. Applied via a second SHA-256 pass over the
        base hash, so it cannot collide with document content.
    :param _legacy_salt: Deprecated. For migration only. Reproduces the old
        (collision-prone) behaviour where salt terms were appended directly to the
        document list. Pass the same list that was formerly concatenated onto the
        documents argument to recover legacy hashes.
    :returns: A SHA-256 hash representing the combined content, in hexadecimal format.
    """
    if _legacy_salt is not None:
        warnings.warn(
            "_legacy_salt is deprecated and exists only for migration purposes. "
            "Switch to the `salt` parameter to avoid hash collisions.",
            DeprecationWarning,
            stacklevel=2,
        )
        return compute_hash(instructions, list(documents) + _legacy_salt)
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

    base_hash = hasher.hexdigest()
    if salt is not None:
        # Combine with salt and re-hash to produce final hash
        salted_hasher = hashlib.sha256()
        salted_hasher.update(base_hash.encode("utf-8"))
        salted_hasher.update(str(salt).encode("utf-8"))
        return salted_hasher.hexdigest()
    else:
        return base_hash
