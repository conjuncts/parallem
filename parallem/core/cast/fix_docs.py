from typing import List, Optional, Union

from parallem.types import FunctionCallRequest, LLMDocument, LLMResponse


def _to_assistant_message(resp: LLMResponse) -> LLMDocument:
    """Converts LLMResponse back into a LLMDocument."""
    val = resp.final_answer
    if resp._pr and resp._pr.function_calls:
        return FunctionCallRequest(
            text_content=val,
            calls=resp.function_calls,
            call_id=resp.call_id,
        )
    return ("assistant", val)


def reduce_to_list(
    documents: Union[Union[LLMDocument, LLMResponse], List[Union[LLMDocument, LLMResponse]]],
    additional_documents: Optional[List[Union[LLMDocument, LLMResponse]]] = None,
) -> List[Union[LLMDocument, LLMResponse]]:
    """
    Turn single or many documents, possibly with additional documents, into always a list.
    """
    if not isinstance(documents, list):
        documents = [documents]
    if additional_documents is None:
        additional_documents = []
    result = documents + additional_documents
    return result


def cast_documents(
    documents: List[Union[LLMDocument, LLMResponse]],
) -> List[LLMDocument]:
    """
    Cast a list of documents to standardize to a common type.

    Turns LLMResponses into LLMDocuments.
    """
    result = documents.copy()
    for i, item in enumerate(documents):
        if isinstance(item, LLMResponse):
            result[i] = _to_assistant_message(item)
    return result
