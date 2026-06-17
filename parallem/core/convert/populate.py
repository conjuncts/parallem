from typing import TypeVar, Union

from parallem.core.exception import IntegrityError
from parallem.core.state.msg_state import MessageState
from parallem.core.response import PendingLLMResponse, ReadyLLMResponse
from parallem.types import BaseRetriever, LLMDocument, LLMResponse

T = TypeVar("T", bound=Union[LLMResponse, LLMDocument])
def populate_llm_response(
    response: T,
    backer: "BaseRetriever",
) -> T:
    """
    Populate an LLMResponse object with any missing information.
    """
    if isinstance(response, PendingLLMResponse):
        response._backer = backer
    elif isinstance(response, ReadyLLMResponse):
        return populate_ready_llm_response(response, backer)
    return response


def populate_ready_llm_response(
    response: T,
    backer: "BaseRetriever",
) -> T:
    """
    Populate an LLMResponse object with any missing information.
    """
    # Implement hydration logic here
    if response._value is None:
        parsed_response = backer.retrieve(response.call_id)
        if parsed_response is None:
            raise IntegrityError("Cached value is no longer available")
        response._value = parsed_response.text
        response._pr = parsed_response
    return response


def populate_msg_state(
    msg_state: MessageState,
    backend: "BaseRetriever",
) -> MessageState:
    """
    Populate MessageState object's LLMResponse items.
    """
    for i, msg in enumerate(msg_state):
        if isinstance(msg, LLMResponse):
            msg_state[i] = populate_llm_response(msg, backend)
    return msg_state
