import json
from typing import Optional, Union
from io import BytesIO
from pipelinellm.core.exception import IntegrityError
from pipelinellm.core.response import PendingLLMResponse
from pipelinellm.types import (
    BaseRetriever,
    DocumentType,
    LLMDocument,
    FunctionCallOutput,
    FunctionCallRequest,
    LLMResponse,
    to_serial_id,
    undo_serial_id,
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
        return (
            str(doc.content).encode("utf-8"),
            "function_call_output",
            json.dumps({"c": doc.call_id, "n": doc.name}, separators=(",", ":")),
        )
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
    elif doc is None or isinstance(doc, (int, float, bool, dict, list)):
        return json.dumps(doc).encode("utf-8"), "json", None
    else:
        raise NotImplementedError(f"Unknown document type: {type(doc)}")


def cast_bytes_to_document(
    doc_value: bytes,
    doc_type: DocumentType,
    doc_extra: Optional[str],
    *,
    retriever: Optional["BaseRetriever"] = None,
) -> Union[LLMDocument, LLMResponse]:
    """
    Convert bytes back to a document, reversing cast_document_to_bytes.

    For ``"function_call"`` and ``"llm_response"`` types, an additional retriever is required
    to hydrate the document with full information.

    :param doc_value: The bytes representing the document content.
    :param doc_type: The document type string.
    :param doc_extra: Additional metadata (role for text tuples, call_id for function_call_output).
    :param retriever: The retriever to use for hydrating the document.
    :return: The reconstructed document.
    :raises NotImplementedError: For unknown types.
    """
    if doc_type == "text":
        text = doc_value.decode("utf-8")
        if doc_extra is not None:
            return (doc_extra, text)
        return text
    elif doc_type == "image":
        return _load_image(doc_value)
    elif doc_type == "function_call_output":
        extra = json.loads(doc_extra) if doc_extra else {}
        return FunctionCallOutput(
            content=doc_value.decode("utf-8"),
            call_id=extra.get("c", ""),
            name=extra.get("n", ""),
        )
    elif doc_type == "function_call":
        short_call_id = undo_serial_id(doc_value.decode("utf-8"))
        if retriever is None:
            raise ValueError("Retriever is required to hydrate FunctionCallRequest")

        full_call_id = retriever.populate_call_id(short_call_id)
        req = retriever.retrieve(full_call_id)
        if not req:
            raise IntegrityError(
                f"Expected FunctionCallRequest for call_id {full_call_id}, got {type(req)}"
            )
        return FunctionCallRequest(
            text_content=req.text, calls=req.function_calls, call_id=full_call_id
        )

    elif doc_type == "llm_response":
        short_call_id = undo_serial_id(doc_value.decode("utf-8"))
        if retriever is None:
            raise ValueError("Retriever is required to hydrate LLMResponse")

        full_call_id = retriever.populate_call_id(short_call_id)
        return PendingLLMResponse(call_id=full_call_id, backend=retriever)
    elif doc_type == "json":
        return json.loads(doc_value.decode("utf-8"))
    else:
        raise NotImplementedError(f"Unknown document type: {doc_type!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_image(data: bytes):
    """Load a PIL Image from raw bytes."""
    from PIL import Image

    return Image.open(BytesIO(data))
