from typing import Union
from io import BytesIO
from pipelinellm.types import (
    DocumentType,
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
        return doc.content.encode("utf-8"), "function_call_output", doc.call_id
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
