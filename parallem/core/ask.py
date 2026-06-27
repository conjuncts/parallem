from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

from parallem.types import (
    FunctionCallOutput,
    HashByOption,
    HumanResponse,
    LLMDocument,
    LLMIdentity,
    LLMResponse,
    ServerTool,
)

if TYPE_CHECKING:
    from parallem.core.state.msg_state import MessageState
    from pydantic import BaseModel


def _raise_exception(*args, **kwargs):
    """
    A default function can be provided to ask_functions, which is executed if the LLM requests
    a function that is not provided. The default behavior raises an exception.
    """
    raise ValueError(
        "LLM asked for a function that was not provided. "
        "Provide a default function via ask_functions(default=lambda x: ...) "
        "or set default=None to ignore."
    )


class Askable(ABC):
    """
    A component that can be asked a question.
    """

    @abstractmethod
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
        structured_output: Optional["BaseModel"] = None,
        tools: Optional[list[Union[dict, ServerTool]]] = None,
        save_input: Optional[bool] = None,
        salt: Optional[str] = None,
        hash_by: List[HashByOption] = ["llm"],
        tag: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Ask the LLM a question.

        :param documents: Documents to use, such as the prompt.
            Can be strings or images.
        :param instructions: The system prompt to use.
        :param llm: The identity of the LLM to use.
            Can be helpful multi-agent or multi-model scenarios.
        :param structured_output: Schema or format specification for structured output.
            For OpenAI: uses structured output via responses.parse().
            For Google: sets response_mime_type and response_schema.
            For Anthropic: sets output_config.format.
        :param tools: A list of tools to make available to the LLM.
            Both user-defined tools (function tools) and server-defined tools
            (ie. web search, code interpreter) are supported.
        :param save_input: Whether to save input documents. Default None: no saving.
        :param salt: A value to include in the hash for differentiation.
        :param hash_by: The names of additional terms to include in the hash for differentiation.
            Example: "llm" includes the LLM name.
        :param tag: An optional tag to associate with the request.
        :returns: A LLMResponse. The value is **lazy loaded**: for best efficiency,
            it should not be resolved until you actually need it.
        """

    @abstractmethod
    def ask_functions(
        self,
        response: Optional[LLMResponse] = None,
        functions: Dict[str, Callable] = None,
        *,
        default: Optional[Callable] = _raise_exception,
        convert_to_str=True,
        **kwargs,
    ) -> List[FunctionCallOutput]:
        """
        If the agent requested any function calls, then this method actually calls user-defined functions.

        Functions should be provided as kwargs.

        :param functions: Available functions to the model. Mapping from function name to callable.
        :param kwargs: Any additional functions will be added to "functions".
        :param default: What to do if a function is not found.
            If Callable, it will be called with the function name and arguments.
            If None, it will be silently ignored.
            Default: an exception will be raised.
        :param convert_to_str: Most APIs (OpenAI, Google, Anthropic) expect function arguments to be strings.
            If True, this method will convert non-string arguments to strings.
        :param cache: If True, cache and reuse function outputs for matching responses.
        :param salt: Optional salt to distinguish cached function-output lookups.
        """

    @abstractmethod
    def ask_human(
        self,
        prompt: str,
        documents: Union[
            LLMDocument,
            LLMResponse,
            List[Union[LLMDocument, LLMResponse]],
            "MessageState",
            None,
        ],
        *additional_documents: LLMDocument,
        salt: Optional[str] = None,
        input_fn: Optional[Callable[[str], str]] = None,
    ) -> HumanResponse:
        """
        Ask a human for input.

        :param prompt: Prompt shown to the human.
        :param documents: Documents used as hash basis (analogous to ask_llm input).
        :param additional_documents: Additional documents appended to ``documents``.
        :param salt: Optional salt to differentiate repeated prompts.
        :param input_fn: Optional callable used to collect human input.
            Defaults to built-in ``input``.
        :returns: A human-provided response.
        """
