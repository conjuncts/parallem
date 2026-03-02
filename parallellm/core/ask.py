from abc import ABC
from typing import TYPE_CHECKING, List, Optional, Union

from parallellm.types import (
    HashByOptions,
    LLMDocument,
    LLMIdentity,
    LLMResponse,
    ServerTool,
)

if TYPE_CHECKING:
    from parallellm.core.state.msg_state import MessageState


class Askable(ABC):
    """
    A component that can be asked a question.
    """

    def ask_llm(
        self,
        documents: Union[
            LLMDocument,
            LLMResponse,
            List[Union[LLMDocument, LLMResponse]],
            "MessageState",
        ],
        *additional_documents: LLMDocument,
        instructions: Optional[str] = None,
        llm: Union[LLMIdentity, str, None] = None,
        salt: Optional[str] = None,
        hash_by: HashByOptions = None,
        text_format: Optional[str] = None,
        tools: Optional[list[Union[dict, ServerTool]]] = None,
        tag: Optional[str] = None,
        save_input: Optional[bool] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Ask the LLM a question.

        :param documents: Documents to use, such as the prompt.
            Can be strings or images.
        :param instructions: The system prompt to use.
        :param llm: The identity of the LLM to use.
            Can be helpful multi-agent or multi-model scenarios.
        :param salt: A value to include in the hash for differentiation.
        :param hash_by: The names of additional terms to include in the hash for differentiation.
            Example: "llm" will also include the LLM name.
        :param text_format: Schema or format specification for structured output.
            For OpenAI: uses structured output via responses.parse().
            For Google: sets response_mime_type and response_schema.
            For Anthropic: not supported.
        :param tools: A list of tools to make available to the LLM.
            Both user-defined tools (function tools) and server-defined tools
            (ie. web search, code interpreter) are supported.
        :param tag: An optional tag to associate with the request.
        :param save_input: Whether to save input documents. Default None: no saving.
        :returns: A LLMResponse. The value is **lazy loaded**: for best efficiency,
            it should not be resolved until you actually need it.
        """
        raise NotImplementedError()
