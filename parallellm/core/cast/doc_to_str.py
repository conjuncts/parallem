from typing import Union
from parallellm.types import (
    DocumentType,
    LLMDocument,
    FunctionCallOutput,
    FunctionCallRequest,
    LLMResponse,
    to_serial_id,
)


def cast_document_to_str(
    doc: Union[LLMDocument, LLMResponse],
) -> tuple[str, DocumentType, str]:
    """
    Convert Document to string, for purpose of serialization.

    :param doc: The LLMDocument to cast
    :return: Tuple of (doc_value: str, doc_type: str, doc_extra: str)
    """
    if isinstance(doc, str):
        return doc, "text", None
    elif isinstance(doc, FunctionCallRequest):
        return to_serial_id(doc.call_id), "function_call", None
    elif isinstance(doc, FunctionCallOutput):
        return doc.content, "function_call_output", doc.call_id
    elif isinstance(doc, tuple):
        return doc[1], "text", doc[0]
    elif isinstance(doc, LLMResponse):
        # Serialize based on LLMResponse ID rather than content
        serial_id = to_serial_id(doc.call_id)
        return serial_id, "llm_response", None
    else:
        raise NotImplementedError(f"Unknown document type: {type(doc)}")
