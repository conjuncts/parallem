from abc import ABC
from dataclasses import dataclass
from typing import (
    Any,
    List,
    Literal,
    TypeAlias,
    TypedDict,
    Optional,
    Union,
    Tuple,
)
from typing_extensions import deprecated
import json

from PIL import Image

from parallem.utils.hardcoded import guess_provider_and_name


class WorkingMetadata(TypedDict):
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


def undo_serial_id(serial_id: str):
    agent_name, seq_id, session_id = serial_id.rsplit(":", 2)
    return {
        "agent_name": agent_name,
        "seq_id": int(seq_id),
        "session_id": int(session_id),
    }


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


HashByOptions: TypeAlias = Optional[list[Literal["llm"]]]


class AskParameters(TypedDict):
    """Parameters for ask_llm()."""

    hash_by: HashByOptions
    save_input: bool
    llm: Union["LLMIdentity", str]


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

    content: Any
    """The output content from the function call."""

    call_id: str
    """The ID of the function call this output corresponds to."""

    name: str
    """The name of the function call this output corresponds to."""

    def __repr__(self):
        return f"FunctionCallOutput(name={self.name}, call_id={(self.call_id or '')[:8]}, content={str(self.content)[:20]}...)"

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


BuiltinProviderType = Literal["openai", "anthropic", "google"]
"""Built-in provider names recognised without the registry."""

ProviderType = Union[BuiltinProviderType, str]
"""
Provider type string.  Built-in values are ``"openai"``, ``"anthropic"``, and
``"google"``.  Arbitrary strings are valid for providers registered via
:func:`parallem.registry.register_provider`.
"""


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

        self.nickname = None
        if self.identity != self.model_name:
            self.nickname = self.identity

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

    def __repr__(self):
        return (
            f"LLMIdentity({self.provider_type}/{self.model_name}"
            + (f' "{self.nickname}"' if self.nickname else "")
            + ")"
        )


class CommonQueryParameters(TypedDict):
    """
    Common parameters for LLM calls across providers. For internal use.
    """

    instructions: Optional[str]
    strict_documents: List[LLMDocument]
    llm: LLMIdentity
    structured_output: Optional[Any]
    tools: Optional[List[dict]]


class MinorTweaks(TypedDict, total=False):
    """
    Minor tweaks for ParaLLeM.
    Holds configs not significant enough to warrant a full keyword argument.
    """

    max_concurrent: Optional[int] = 20
    "Maximum number of concurrent tasks in ConcurrentBackend."

    batch_user_confirmation: bool = True
    "Whether to ask for user confirmation before submitting a batch."

    batch_wait_until_complete: bool = False
    "Whether to wait for all batches to complete before proceeding."

    batch_max_size: int = 1000
    "Maximum number of calls submitted per provider batch request."

    batch_input_format: Literal["jsonl", "zip"] = "zip"
    "Whether to compress batch inputs."

    batch_output_format: Literal["jsonl", "zip"] = "zip"
    "Whether to compress batch outputs."


class LLMResponse:
    """
    Any response outputted by an LLM. **You must access the value through `final_answer`.**
    """

    def __init__(self, value: str, *, call_id: CallIdentifier = None):
        self._value = value
        self.call_id = call_id
        self._pr: Optional[ParsedResponse] = None

    @property
    def final_answer(self) -> str:
        """
        Returns the text within this response.

        :returns: The resolved string response. If this value is not available,
            execution should stop gracefully and proceed to the next batch.
        """
        return self._value

    @property
    def final_json(self) -> Optional[dict]:
        """
        Returns the response loaded as a dictionary. Returns None if invalid.
        """
        try:
            return json.loads(self.final_answer)
        except json.JSONDecodeError:
            return None

    @property
    def function_calls(self) -> list[FunctionCall]:
        """
        Returns function calls to user-defined functions.
        """
        if self._pr and self._pr.function_calls:
            # cast and jsonify if needed
            return self._pr.function_calls
        return []

    def __repr__(self):
        v = self._value
        if v and len(v) > 50:
            v = v[:47] + "..."
        return (
            f"{self.__class__.__name__}({v!r}, doc_hash={self.call_id['doc_hash'][:8]})"
        )

    def __str__(self):
        if self._value is not None:
            return self._value
        return repr(self)

    def __await__(self):
        async def _sync_await_response():
            return self.final_answer

        return _sync_await_response().__await__()

    @deprecated("Use final_answer property instead.")
    def resolve(self) -> str:
        return self.final_answer

    @deprecated("Use final_json property instead.")
    def resolve_json(self) -> Optional[dict]:
        return self.final_json

    @deprecated("Use function_calls property instead.")
    def resolve_function_calls(self) -> list[FunctionCall]:
        """
        Resolves response, then returns function calls to user-defined functions.
        """
        return self.function_calls

    @property
    def output_text(self) -> str:
        "Alias for final_answer - OpenAI compliant."
        return self.final_answer

    @property
    def output_parsed(self) -> Optional[dict]:
        "Alias for final_json - OpenAI compliant."
        return self.final_json


class HumanResponse(LLMResponse):
    """
    A response provided by a human. Useful for human-in-the-loop.

    If created without a value, ``final_answer`` lazily loads it from the retriever
    using ``origin_type=1``.
    """

    def __init__(
        self,
        value: Optional[str],
        *,
        call_id: CallIdentifier = None,
        backend: Optional["BaseRetriever"] = None,
    ):
        super().__init__(value=value, call_id=call_id)
        self._backend = backend

    @property
    def final_answer(self) -> Optional[str]:
        if self._value is not None:
            return self._value

        if self._backend is None or self.call_id is None:
            return self._value

        pr = self._backend.retrieve(self.call_id, origin_type=1)
        if pr is None:
            return None

        self._pr = pr
        self._value = pr.text
        return self._value


class BaseRetriever(ABC):
    """
    Class where retrieve(call_id) and populate_call_id(call_id) is defined
    """

    def retrieve(
        self,
        call_id: CallIdentifier,
        metadata=False,
        *,
        origin_type: Optional[int] = None,
    ) -> Optional[ParsedResponse]:
        """
        Retrieve a response.

        :param call_id: The task identifier containing agent_name, doc_hash, and seq_id.
        :param origin_type: Optional origin marker filter. ``None`` retrieves only
            LLM-originated rows, ``1`` retrieves only human-originated rows.
        :returns: The retrieved ParsedResponse.
        """
        raise NotImplementedError

    async def await_response(
        self, call_id: CallIdentifier, metadata: bool = False
    ) -> Optional[ParsedResponse]:
        # return await asyncio.to_thread(self.retrieve, call_id, metadata)
        return self.retrieve(call_id, metadata=metadata)

    def populate_call_id(
        self, call_id: CallIdentifier, *, metadata=False
    ) -> CallIdentifier:
        """
        Given a call_id with potentially missing fields (like doc_hash), populate those fields based on the backend's data.

        :param call_id: The input CallIdentifier with some fields potentially missing.
        :param metadata: Whether to include metadata in the populated call_id.
        :returns: A fully populated CallIdentifier with all necessary fields filled in.
        """
        raise NotImplementedError
