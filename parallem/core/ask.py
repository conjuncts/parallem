from abc import ABC
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Union

from parallem.types import (
    FunctionCallOutput,
    HashByOptions,
    HumanResponse,
    LLMDocument,
    LLMIdentity,
    LLMResponse,
    ServerTool,
)

if TYPE_CHECKING:
    from parallem.core.state.msg_state import MessageState
    from pydantic import BaseModel


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
        structured_output: Optional["BaseModel"] = None,
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
        :param structured_output: Schema or format specification for structured output.
            For OpenAI: uses structured output via responses.parse().
            For Google: sets response_mime_type and response_schema.
            For Anthropic: sets output_config.format.
        :param tools: A list of tools to make available to the LLM.
            Both user-defined tools (function tools) and server-defined tools
            (ie. web search, code interpreter) are supported.
        :param tag: An optional tag to associate with the request.
        :param save_input: Whether to save input documents. Default None: no saving.
        :returns: A LLMResponse. The value is **lazy loaded**: for best efficiency,
            it should not be resolved until you actually need it.
        """
        raise NotImplementedError()

    def ask_functions(
        self,
        response: Optional[LLMResponse] = None,
        functions: Dict[str, Callable] = None,
        *,
        subagent_names: Optional[Sequence[str]] = None,
        if_func_not_exist: Union[str, Exception, None] = None,
        **kwargs,
    ) -> List[FunctionCallOutput]:
        """
        If the agent requested any function calls, then this method actually calls user-defined functions.

        Functions should be provided as kwargs.

        :param functions: Available functions to the model. Mapping from function name to callable.
        :param kwargs: Any additional functions will be added to "functions".
        :param if_func_not_exist: What to do if a function is not found.
            If an Exception is passed, it will be raised.
            If a string is passed, it will be added to the conversation as an error message
            but allowed to continue.
            If None, it will be silently ignored.
            Default: None.
        """
        raise NotImplementedError()

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
        raise NotImplementedError()
