from abc import ABC
from dataclasses import dataclass
from typing import (
    List,
    Literal,
    TypedDict,
    Optional,
    Union,
    Tuple,
)
import json

from PIL import Image

from parallellm.utils.hardcoded import guess_provider_and_name


class AgentMetadata(TypedDict):
    """Metadata for an agent"""

    pass


class WorkingMetadata(TypedDict):
    agents: dict[str, AgentMetadata]

    session_counter: int
    """
    Numeric ID of session. Increments on each new BatchManager creation.
    """


class CallMetadata(TypedDict):
    """Not required metadata for a call."""

    provider_type: Optional[str]
    """Specific provider type (ie. openai, anthropic, etc)"""
    tag: Optional[str]
    """An optional tag to associate with the call."""


class CallIdentifier(TypedDict):
    agent_name: str
    """Name of the agent (if any, otherwise default) making the call."""
    doc_hash: str
    seq_id: int
    session_id: int
    """Numeric ID of session. For tracking/metadata purposes only."""

    meta: Optional[CallMetadata]


def to_serial_id(call_id: CallIdentifier, *, add_sess=True) -> str:
    if not add_sess:
        return f"{call_id['agent_name']}:{call_id['seq_id']}"
    return f"{call_id['agent_name']}:{call_id['seq_id']}:{call_id['session_id']}"


@dataclass
class BatchIdentifier:
    call_ids: List[CallIdentifier]

    custom_ids: List[str]
    """Custom IDs assigned to each call in the batch, in the same order as call_ids."""

    batch_uuid: str
    """A unique identifier given for the batch by the provider."""


@dataclass
class CohortIdentifier:
    """
    If a batch is very large (>1000), it may have to be split into multiple sub-batches
    to keep below file size limits.

    A cohort is a collection of such sub-batches that together represent a single logical batch.
    """

    batch_ids: List[BatchIdentifier]

    session_id: int
    """Numeric ID of session. Here, it also serves as a cohort ID."""


HashByOptions = Optional[list[Literal["llm"]]]


class AskParameters(TypedDict):
    """Parameters for ask_llm()."""

    hash_by: HashByOptions
    save_input: bool


BatchStatus = Literal["ready", "error"]


@dataclass(slots=True)
class BatchResult:
    status: BatchStatus

    raw_output: Optional[str]
    """The raw output from the provider (for debugging/logging purposes)."""

    parsed_responses: Optional[List["ParsedResponse"]]
    """List of parsed responses, if available."""


class FunctionCall:
    """Represents a single tool call to a user-defined function."""

    __slots__ = ("name", "call_id", "args", "arg_str")

    def __init__(
        self,
        name: str,
        arguments: Union[str, dict],
        call_id: str,
    ):
        """
        Initialize a FunctionCall.

        Args:
            name: The name of the function being called.
            call_id: The unique identifier for this function call.
            arguments: The arguments for the function call as a dictionary.
            arg_str: The arguments for the function call as a JSON string.
        """
        self.name = name
        self.call_id = call_id

        if isinstance(arguments, str):
            # Parse arg_str to arguments
            self.args = json.loads(arguments)
        else:
            self.args = arguments
        self.arg_str = json.dumps(self.args)

    def __iter__(self):
        """Allow unpacking into tuple for backward compatibility."""
        return iter((self.name, self.args, self.call_id))

    def __repr__(self):
        return f"FunctionCall(name={self.name}, call_id={(self.call_id or '')[:8]}, args={self.args})"

    def __str__(self):
        return self.__repr__()


@dataclass(slots=True)
class FunctionCallRequest:
    """Represents the LLM requesting function/tool call(s)"""

    text_content: str
    """Text content, like thoughts about invoking a function."""

    calls: List[FunctionCall]
    """List of function calls."""

    call_id: CallIdentifier

    def __repr__(self):
        brief_calls = [
            f"{call.name}({(call.call_id or '')[:8]})" for call in self.calls
        ]
        return f"FunctionCallRequest(text_content={self.text_content}, calls={brief_calls})"

    def __str__(self):
        return self.__repr__()


@dataclass(slots=True)
class FunctionCallOutput:
    """Represents the output/result of a function/tool call."""

    content: str
    """The output content from the function call."""

    call_id: str
    """The ID of the function call this output corresponds to."""

    name: str
    """The name of the function call this output corresponds to."""

    def __repr__(self):
        return f"FunctionCallOutput(name={self.name}, call_id={(self.call_id or '')[:8]}, content={self.content[:20]}...)"

    def __str__(self):
        return self.__repr__()


ServerToolType = Literal["web_search", "code_interpreter"]


class ServerTool(ABC):
    """
    Represents a tool implemented by the server, such as web search or code interpreter.
    """

    server_tool_type: ServerToolType
    """Must be set by subclasses to identify the provider type."""

    kwargs: dict
    """Extra custom kwargs"""


LLMDocument = Union[
    str,
    Image.Image,
    Tuple[Literal["user", "assistant", "system", "developer"], str],
    FunctionCallRequest,
    FunctionCallOutput,
]
"""
Type alias for documents that can be either text or images.
"""

DocumentType = Literal["text", "function_call", "function_call_output", "llm_response"]
"""
Enum for valid document types. Closely matches OpenAI's document types.
LLMResponse: Any response from the LLM.
"""


ProviderType = Literal["openai", "anthropic", "google"]


@dataclass(slots=True)
class ParsedResponse:
    """
    Represents a parsed response from an LLM provider.

    This dataclass encapsulates the three key components of a parsed response:
    - The response text content
    - The response ID (if available from the provider)
    - Additional metadata from the provider
    """

    text: str
    """The main text content of the response."""

    response_id: Optional[str]
    """
    The unique identifier for this response from the provider.
    Only populated for fresh responses; will be None for cached response.
    """

    metadata: Optional[dict]
    """Additional metadata from the provider (usage stats, model info, etc.)."""

    function_calls: Optional[List[FunctionCall]] = None

    custom_id: Optional[str] = None
    """The unique identifier for this response, only populated in batch requests."""

    old_session_id: Optional[int] = None
    """
    The session_id during which this response was originally generated.
    Only populated for cached responses; will be None for fresh responses.
    """

    old_seq_id: Optional[int] = None
    """
    The seq_id during which this response was originally generated.
    Only populated for cached responses; will be None for fresh responses.
    """

    error_code: Optional[int] = None
    """If successful, should be None."""


@dataclass(slots=True)
class ParsedError(ParsedResponse):
    """
    Represents a parsed error from an LLM provider.
    """

    error_code: int = 0
    """Indicates this response represents an error, and contains the error code (e.g. 429, 500, etc.)"""


class LLMIdentity:
    def __init__(
        self,
        identity: str,
        *,
        provider_type: Optional[ProviderType] = None,
        model_name: Optional[str] = None,
    ):
        """
        Identify a specific LLM agent.

        :param identity: The identity string of the LLM.
            Can be a canonical name, like "gpt-4o-mini", or a convenient
            nickname like "alex".
        :param provider: The provider of the LLM, if known. For instance, "openai".
        :param model_name: If a nickname is used for identity, the actual model name.
        """
        self.identity = identity

        if provider_type is None:
            # do some guessing
            provider_type, model_name = guess_provider_and_name(identity)
            if provider_type is None:
                raise ValueError(
                    f"Unknown provider for identity '{identity}'. Please specify provider explicitly."
                )
        elif model_name is None:
            # if provider is given but not model_name, assume identity is model_name
            model_name = identity
        # else: both provider and model_name are given, use as-is

        self.provider_type = provider_type
        self.model_name = model_name

    def __hash__(self):
        """Make LLMIdentity hashable based on provider and model_name."""
        return hash((self.provider_type, self.model_name, self.identity))

    def __eq__(self, other):
        """Compare LLMIdentity instances based on provider and model_name."""
        if not isinstance(other, LLMIdentity):
            return False
        return (
            self.provider_type == other.provider_type
            and self.model_name == other.model_name
            and self.identity == other.identity
        )


class CommonQueryParameters(TypedDict):
    """
    Common parameters for LLM calls across providers. For internal use.
    """

    instructions: Optional[str]
    strict_documents: List[LLMDocument]
    llm: LLMIdentity
    text_format: Optional[dict]
    tools: Optional[List[dict]]


@dataclass(frozen=True, slots=True)
class MinorTweaks:
    """
    Minor tweaks for the ParallelLLM framework.
    Holds configs not significant enough to warrant a full keyword argument.
    """

    max_concurrent: Optional[int] = 20
    "Maximum number of concurrent tasks in ConcurrentBackend."

    batch_user_confirmation: bool = True
    "Whether to ask for user confirmation before submitting a batch."

    batch_wait_until_complete: bool = True
    "Whether to wait for all batches to complete before proceeding."


class LLMResponse:
    """
    Any response outputted by an LLM. **You must call resolve() to obtain the final value.**
    """

    def __init__(self, value: str, *, call_id: CallIdentifier = None):
        self.value = value
        self.call_id = call_id
        self._pr: Optional[ParsedResponse] = None

    def resolve(self) -> str:
        """
        Resolve the response to a string.

        :returns: The resolved string response. If this value is not available,
            execution should stop gracefully and proceed to the next batch.
        """
        return self.value

    def resolve_json(self) -> Optional[dict]:
        """
        Resolve the response and automatically convert it to JSON. Returns None if invalid.

        :param self: Description
        :return: Description
        :rtype: dict
        """
        try:
            return json.loads(self.value)
        except json.JSONDecodeError:
            return None

    def resolve_function_calls(self) -> list[FunctionCall]:
        """
        Resolve function calls (tool calls to user-defined functions) associated with this response.

        :param to_dict: Whether to parse the function calls' arguments into dictionaries (if they're JSON strings)
        :returns: A list of FunctionCall objects
        """
        if self._pr and self._pr.function_calls:
            # cast and jsonify if needed
            return self._pr.function_calls
        return []

    def __repr__(self):
        v = self.value
        if v and len(v) > 50:
            v = v[:47] + "..."
        return (
            f"{self.__class__.__name__}({v!r}, doc_hash={self.call_id['doc_hash'][:8]})"
        )

    def __str__(self):
        if self.value is not None:
            return self.value
        return repr(self)


class HumanResponse(LLMResponse):
    """
    A response provided by a human. Useful for human-in-the-loop.

    Like LLMResponse, you must call resolve() to obtain the final value. TODO.
    """
