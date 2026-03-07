from pipelinellm.core.backend import BaseBackend
from pipelinellm.core.exception import IntegrityError
from pipelinellm.core.state.msg_state import MessageState
from pipelinellm.core.response import PendingLLMResponse, ReadyLLMResponse
from pipelinellm.types import LLMResponse


def hydrate_llm_response(
    response: LLMResponse,
    backend: "BaseBackend",
) -> LLMResponse:
    """
    Hydrate an LLMResponse object with any missing information.
    """
    if isinstance(response, PendingLLMResponse):
        response._backend = backend
    elif isinstance(response, ReadyLLMResponse):
        return hydrate_ready_llm_response(response, backend)
    return response


def hydrate_ready_llm_response(
    response: ReadyLLMResponse,
    backend: "BaseBackend",
) -> ReadyLLMResponse:
    """
    Hydrate an LLMResponse object with any missing information.
    """
    # Implement hydration logic here
    if response.value is None:
        parsed_response = backend.retrieve(response.call_id)
        if parsed_response is None:
            raise IntegrityError("Cached value is no longer available")
        response.value = parsed_response.text
        response._pr = parsed_response
    return response


def hydrate_msg_state(
    msg_state: MessageState,
    backend: "BaseBackend",
) -> MessageState:
    """
    Hydrate a MessageState object with any missing information.
    """
    for i, msg in enumerate(msg_state):
        if isinstance(msg, LLMResponse):
            msg_state[i] = hydrate_llm_response(msg, backend)
    return msg_state
