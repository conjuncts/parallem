import base64
import json
from typing import Optional, Union
from io import BytesIO
from pipelinellm.types import (
    DocumentType,
    FunctionCall,
    LLMDocument,
    FunctionCallOutput,
    FunctionCallRequest,
    LLMResponse,
    to_serial_id,
)
from pipelinellm.utils.image import is_image


def cast_document_to_bytes(
    doc: Union[LLMDocument, LLMResponse],
) -> tuple[bytes, DocumentType, str]:
    """
    Convert Document to bytes, for purpose of serialization.

    :param doc: The LLMDocument to cast
    :return: Tuple of (doc_value: bytes, doc_type: str, doc_extra: str)
    """
    if isinstance(doc, str):
        return doc.encode("utf-8"), "text", None
    elif isinstance(doc, FunctionCallRequest):
        return to_serial_id(doc.call_id).encode("utf-8"), "function_call", None
    elif isinstance(doc, FunctionCallOutput):
        return str(doc.content).encode("utf-8"), "function_call_output", doc.call_id
    elif isinstance(doc, tuple):
        return doc[1].encode("utf-8"), "text", doc[0]
    elif is_image(doc):
        # Dump image as bytes
        buffered = BytesIO()
        doc.save(buffered, format=doc.format or "PNG")
        return buffered.getvalue(), "image", None
    elif isinstance(doc, LLMResponse):
        # Serialize based on LLMResponse ID rather than content
        serial_id = to_serial_id(doc.call_id)
        return serial_id.encode("utf-8"), "llm_response", None
    else:
        raise NotImplementedError(f"Unknown document type: {type(doc)}")


def cast_bytes_to_document(
    doc_value: bytes,
    doc_type: DocumentType,
    doc_extra: Optional[str],
) -> Union[LLMDocument, LLMResponse]:
    """
    Convert bytes back to a document, reversing cast_document_to_bytes for lossless types.

    Note: ``"function_call"`` and ``"llm_response"`` only store the serial ID in
    ``cast_document_to_bytes``, so they cannot be fully reconstructed here.
    Use :func:`serialize_document` / :func:`deserialize_document` for full round-trip.

    :param doc_value: The bytes representing the document content.
    :param doc_type: The document type string.
    :param doc_extra: Additional metadata (role for text tuples, call_id for function_call_output).
    :return: The reconstructed document.
    :raises NotImplementedError: For lossy types (``"function_call"``, ``"llm_response"``)
        or unknown types.
    """
    if doc_type == "text":
        text = doc_value.decode("utf-8")
        if doc_extra is not None:
            return (doc_extra, text)
        return text
    elif doc_type == "image":
        return _load_image(doc_value)
    elif doc_type == "function_call_output":
        return FunctionCallOutput(
            content=doc_value.decode("utf-8"),
            call_id=doc_extra,
            name="",
        )
    elif doc_type == "function_call":
        raise NotImplementedError(
            "FunctionCallRequest is not fully stored by cast_document_to_bytes (lossy). "
            "Use deserialize_document instead."
        )
    elif doc_type == "llm_response":
        raise NotImplementedError(
            "LLMResponse is not fully stored by cast_document_to_bytes (lossy). "
            "Use deserialize_document instead."
        )
    else:
        raise NotImplementedError(f"Unknown document type: {doc_type!r}")


def serialize_document(doc: Union[LLMDocument, LLMResponse]) -> str:
    """
    Fully serialize a document to a JSON string for lossless round-trip storage.

    Unlike :func:`cast_document_to_bytes`, this preserves all fields of every
    document type, including :class:`~pipelinellm.types.FunctionCallRequest` and
    :class:`~pipelinellm.types.LLMResponse`.

    :param doc: The document or response to serialize.
    :return: A JSON string.
    :raises NotImplementedError: For unsupported document types.
    """
    if isinstance(doc, str):
        return json.dumps({"type": "text", "value": doc, "role": None})
    elif isinstance(doc, FunctionCallRequest):
        return json.dumps(
            {
                "type": "function_call",
                "text_content": doc.text_content,
                "calls": [[c.name, c.args, c.call_id] for c in doc.calls],
                "call_id": dict(doc.call_id),
            }
        )
    elif isinstance(doc, FunctionCallOutput):
        return json.dumps(
            {
                "type": "function_call_output",
                "content": str(doc.content),
                "call_id": doc.call_id,
                "name": doc.name,
            }
        )
    elif isinstance(doc, tuple):
        return json.dumps({"type": "text", "value": doc[1], "role": doc[0]})
    elif is_image(doc):
        buffered = BytesIO()
        doc.save(buffered, format=doc.format or "PNG")
        return json.dumps(
            {
                "type": "image",
                "data": base64.b64encode(buffered.getvalue()).decode("ascii"),
                "format": doc.format or "PNG",
            }
        )
    elif isinstance(doc, LLMResponse):
        return json.dumps(
            {
                "type": "llm_response",
                "value": doc.value,
                "call_id": dict(doc.call_id) if doc.call_id else None,
            }
        )
    else:
        raise NotImplementedError(f"Unknown document type: {type(doc)}")


def deserialize_document(data: str) -> Union[LLMDocument, LLMResponse]:
    """
    Deserialize a document from a JSON string produced by :func:`serialize_document`.

    :param data: The JSON string.
    :return: The reconstructed document or response.
    :raises NotImplementedError: For unknown type fields in the JSON.
    """
    obj = json.loads(data)
    doc_type = obj["type"]

    if doc_type == "text":
        role = obj.get("role")
        value = obj["value"]
        if role is not None:
            return (role, value)
        return value
    elif doc_type == "function_call":
        calls = [
            FunctionCall(name=name, arguments=args, call_id=cid)
            for name, args, cid in obj["calls"]
        ]
        return FunctionCallRequest(
            text_content=obj["text_content"],
            calls=calls,
            call_id=obj["call_id"],
        )
    elif doc_type == "function_call_output":
        return FunctionCallOutput(
            content=obj["content"],
            call_id=obj["call_id"],
            name=obj["name"],
        )
    elif doc_type == "image":
        raw = base64.b64decode(obj["data"])
        return _load_image(raw)
    elif doc_type == "llm_response":
        return LLMResponse(value=obj["value"], call_id=obj.get("call_id"))
    else:
        raise NotImplementedError(
            f"Unknown document type in serialized data: {doc_type!r}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_image(data: bytes):
    """Load a PIL Image from raw bytes."""
    from PIL import Image

    return Image.open(BytesIO(data))
