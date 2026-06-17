import inspect
import json
from functools import partial
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
    get_args,
    get_origin,
)
from typing_extensions import deprecated
from parallem.core.ask import Askable
from parallem.core.convert.fix_docs import cast_documents, reduce_to_list
from parallem.core.exception import NotAvailable, PendingNotAvailable
from parallem.core.hash import build_hash_salt_terms, compute_hash
from parallem.core.hydrate import hydrate_msg_state
from parallem.core.memoize.memoize_context import MemoizeContext
from parallem.core.state.msg_state import MessageState
from parallem.core.response import (
    ReadyLLMResponse,
)
from parallem.logging.dash_logger import HashStatus
from parallem.tools.auto_schema import to_tool_schema
from parallem.types import (
    AskParameters,
    CallIdentifier,
    CommonQueryParameters,
    FunctionCallOutput,
    HashByOption,
    HumanResponse,
    LLMDocument,
    LLMIdentity,
    LLMResponse,
    ParsedResponse,
    ServerTool,
)



_FUNCTION_CALL_ORIGIN_TYPE = 2
if TYPE_CHECKING:
    from parallem.core.agent.orchestrator import AgentOrchestrator
    from pydantic import BaseModel


def _is_agent_context_annotation(annotation) -> bool:
    if annotation is inspect.Parameter.empty:
        return False

    if isinstance(annotation, str):
        cleaned = annotation.replace(" ", "")
        return cleaned == "AgentContext" or cleaned.endswith(".AgentContext")

    if hasattr(annotation, "__forward_arg__"):
        return _is_agent_context_annotation(annotation.__forward_arg__)

    origin = get_origin(annotation)
    if origin is not None:
        return any(_is_agent_context_annotation(arg) for arg in get_args(annotation))

    return getattr(annotation, "__name__", None) == "AgentContext"


class AgentContext(Askable):
    """Context manager for one "agent", which in parallem is one single autonomous process.

    parallem does things a bit differently.
    While typically an agent is associated with a single LLM,
    parallem identifies an agent with a process, program, or algorithm
    which itself can ask LLMs questions, but also functions, MCP servers, and humans.
    It just so happens that the agent uses LLM(s) to automate much of its decision making."""

    def __init__(
        self,
        agent_name: str,
        orch: "AgentOrchestrator",
        *,
        ask_params: Optional[AskParameters] = None,
        ignore_cache: bool = False,
        error_mode: Literal["ignore", "emit", "raise"] = "raise",
    ):
        self.agent_name = agent_name
        self._orch = orch

        self.ask_params = ask_params or {}
        if self.ask_params:
            self.ask_llm = self._bind_ask_llm(self.ask_params)
        self.ignore_cache = ignore_cache

        self._msg_state: Optional[MessageState] = None
        "MessageState for this agent. Some pipelines won't use this (so it will be None)."
        self._print_context = None
        self._error_mode = error_mode

    def __enter__(self):
        # Only redirect stdout when the dashboard display is enabled.
        if self._orch._dashlog is not None and self._orch._dashlog.display:
            self._print_context = self._orch._dashlog
            self._print_context.__enter__()
        else:
            self._print_context = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None and exc_type not in (NotAvailable, PendingNotAvailable):
            self._orch._dashlog.agent_errored(self.agent_name)
            if self._error_mode == "emit":
                self._orch._logger.error(type(exc_value).__name__ + ": " + str(exc_value))
        if self._print_context is not None:
            self._print_context.__exit__(exc_type, exc_value, traceback)
            self._print_context = None
        if exc_type in (NotAvailable, PendingNotAvailable):
            # swallow NotAvailable and its subclasses (like PendingNotAvailable)
            return True
        if (
            exc_type is not None
            and self._error_mode != "raise"
            and issubclass(exc_type, Exception) # exclude BaseException
        ):
            return True
        if self._orch.strategy == "batch" and exc_type in (
            NotAvailable,
            PendingNotAvailable,
        ):
            # swallow NotAvailable errors only in batch mode
            return True
        return False

    @deprecated("Use builtin print() directly.")
    def print(self, *args, **kwargs):
        print(*args, **kwargs)

    def _coerce_tools(
        self,
        tools: Optional[list[Union[dict, ServerTool, Callable]]],
    ):
        """
        Coerce tools into a consistent format: list of function call dicts or ServerTools
        Turns Callable into dict using pllm.to_tool_schema
        """
        # coerce callable tools to dict format with name and parameters
        if tools is not None:
            coerced_tools = []

            # if not sequence
            if not isinstance(tools, (list, tuple)):
                tools = [tools]
            for tool in tools:
                if isinstance(tool, (dict, ServerTool)):
                    coerced_tools.append(tool)
                elif callable(tool):
                    coerced_tools.extend(to_tool_schema(tool))
                else:
                    raise ValueError(f"Tool {tool} is not a dict, ServerTool, or callable.")
            return coerced_tools
        return tools

    def _coerce_options(
        self,
        llm,
        structured_output,
        kwargs,
    ):
        """Helper method to ensure LLM options have the right type, coercing fields as needed."""
        # Handle legacy text_format alias
        legacy_text_format = kwargs.pop("text_format", None)
        if structured_output is not None and legacy_text_format is not None:
            raise ValueError(
                "Cannot specify both structured_output and text_format. "
                "text_format is a legacy alias for structured_output."
            )
        if structured_output is None:
            structured_output = legacy_text_format

        if llm is None:
            llm = self._orch._provider.get_default_llm_identity()
        elif isinstance(llm, str):
            best_provider_type = self._orch._provider.provider_type
            if best_provider_type in {"multi"}:
                best_provider_type = None
            llm = LLMIdentity(llm, provider_type=best_provider_type)

        provider_type = self._orch._provider.provider_type
        if provider_type is None:
            provider_type = llm.provider_type

        return llm, provider_type, structured_output

    def _compute_hash(
        self,
        params: CommonQueryParameters,
        *,
        salt,
        hash_by,
        provider_type,
        kwargs=None,
    ):
        """Compute the input hash (doc_hash) for a list of documents."""
        # Compute salt
        salt_terms = build_hash_salt_terms(
            salt=salt,
            hash_by=hash_by,
            llm=params["llm"],
            provider_type=provider_type,
            tools=params["tools"],
            structured_output=params["structured_output"],
            kwargs=kwargs,
        )

        # Use a null-byte separator so individual terms cannot be confused with one
        # another, and pass as the `salt` parameter (applied via re-hash) so that
        # salt content can never collide with document content.
        combined_salt = "\x00".join(salt_terms) if salt_terms else None
        hashed = compute_hash(
            params["instructions"], params["strict_documents"], salt=combined_salt
        )
        return hashed, salt_terms

    def _get_cached_response(
        self,
        call_id,
    ):
        """Helper method to check for and return a cached response, if it exists."""
        cached = None if self.ignore_cache else self._orch._backend.retrieve(call_id)
        if cached is not None:
            # populate the old session_id. This helps make to_serial_id deterministic
            if cached.old_session_id is not None:
                call_id["session_id"] = cached.old_session_id
                call_id["seq_id"] = cached.old_seq_id
            return ReadyLLMResponse(
                call_id=call_id,
                pr=cached,
            )

    def ask_llm(
        self,
        documents: Union[
            LLMDocument,
            LLMResponse,
            List[Union[LLMDocument, LLMResponse]],
            MessageState,
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
        # 1. assign sequential ID, input checks
        seq_id = self._orch.next_seq_id(self.agent_name)

        llm, provider_type, structured_output = self._coerce_options(llm, structured_output, kwargs)
        tools = self._coerce_tools(tools)

        if isinstance(documents, MessageState):
            documents = list(documents)

        documents = reduce_to_list(documents, list(additional_documents))
        resolved_docs = cast_documents(documents)

        # convenient params dict
        params: CommonQueryParameters = {
            "instructions": instructions,
            "strict_documents": resolved_docs,
            "llm": llm,
            "structured_output": structured_output,
            "tools": tools,
        }
        # 2. compute hash for inputs
        hashed, salt_terms = self._compute_hash(
            params,
            salt=salt,
            hash_by=hash_by,
            provider_type=provider_type,
            kwargs=kwargs,
        )

        call_id: CallIdentifier = {
            "agent_name": self.agent_name,
            "doc_hash": hashed,
            "seq_id": seq_id,
            "session_id": self._orch.get_session_counter(),
            "meta": {
                "provider_type": provider_type,
                "tag": tag,
            },
        }

        # 3. save inputs if needed (pass `params` mapping)
        if save_input:
            self._orch._backend.store_input(
                call_id,
                params=params,
                hash_by=hash_by,
                salt=salt,
                request_kwargs=kwargs,
            )

        # 4. use cache if available
        cached = self._get_cached_response(call_id)
        if cached is not None:
            self._orch._dashlog.update_call(call_id, HashStatus.CACHED)
            return cached

        # 5. use API
        if not self._orch._provider.is_compatible(provider_type):
            raise ValueError(f"LLM {llm} is not compatible with provider {provider_type}")

        # The below function calls the LLM
        return self._orch._backend.call_llm(
            self._orch._provider,
            params,
            call_id=call_id,
            **kwargs,
        )

    def _bind_ask_llm(
        self,
        ask_params: Optional[AskParameters] = None,
        **bound_kwargs,
    ) -> Callable[..., LLMResponse]:
        """
        Return a callable version of ``ask_llm`` with bound default keyword arguments.

        Bound defaults are merged first, then per-call kwargs override them.
        """
        merged_defaults = dict(ask_params or {})
        merged_defaults.update(bound_kwargs)

        return partial(AgentContext.ask_llm, self, **merged_defaults)

    def ask_functions(
        self,
        response: LLMResponse,
        functions: Dict[str, Callable] = None,
        *,
        subagent_names: Optional[Sequence[str]] = None,
        if_func_not_exist: Union[str, Exception, None] = None,
        convert_to_str=True,
        cache: bool = False,
        salt: Optional[str] = None,
        **kwargs,
    ) -> List[FunctionCallOutput]:
        if functions is None:
            functions = {}
        functions.update(kwargs)
        if not functions:
            raise ValueError(
                "No functions provided to ask_functions. Provide functions as a dict or as kwargs."
            )

        cache_call_id = None
        if cache:
            if response.call_id is None:
                raise ValueError("ask_functions(cache=True) requires response.call_id to be set")

            cache_call_id = {
                "agent_name": self.agent_name,
                "doc_hash": response.call_id["doc_hash"],
                "seq_id": self._orch.next_seq_id(self.agent_name),
                "session_id": self._orch.get_session_counter(),
                "meta": response.call_id.get("meta"),
            }
            if salt is not None:
                cache_call_id["doc_hash"] = compute_hash(cache_call_id["doc_hash"], [], salt=salt)

            datastore = self._orch._backend._get_datastore()
            cached = None if self.ignore_cache else datastore.retrieve(cache_call_id, origin_type=_FUNCTION_CALL_ORIGIN_TYPE)
            if cached is not None:
                if cached.old_session_id is not None:
                    cache_call_id["session_id"] = cached.old_session_id
                    cache_call_id["seq_id"] = cached.old_seq_id
                return self._deserialize_function_call_outputs(cached.text)

        subagent_name_iter = iter(subagent_names) if subagent_names is not None else None

        # Check if response has function calls. If so, delegate to user-defined functions.
        fcs = response.function_calls
        fc_outs = []
        for fc in fcs:
            callme = functions.get(fc.name)
            if callme is None:
                # Function not found
                if if_func_not_exist is ValueError:
                    raise ValueError(f"LLM asked for {fc.name}, but it was not provided.")
                if isinstance(if_func_not_exist, Exception):
                    raise if_func_not_exist
                elif if_func_not_exist is None:
                    continue
                else:
                    fc_outs.append(if_func_not_exist)
                    continue

            call_args = dict(fc.args)
            call_args = self._inject_subagent_context(
                callme,
                call_args,
                subagent_name_iter,
            )

            # Execute the function
            result = callme(**call_args)
            if convert_to_str and not isinstance(result, str):
                result = str(result)
            fc_outs.append(FunctionCallOutput(content=result, name=fc.name, fcall_id=fc.fcall_id))

        if cache and not self.ignore_cache and cache_call_id is not None:
            datastore = self._orch._backend._get_datastore()
            datastore.store(
                cache_call_id,
                ParsedResponse(
                    text=self._serialize_function_call_outputs(fc_outs),
                    response_id=None,
                    metadata=None,
                    function_calls=None,
                ),
                origin_type=_FUNCTION_CALL_ORIGIN_TYPE,
            )
        return fc_outs

    def _serialize_function_call_outputs(self, outputs: List[FunctionCallOutput]) -> str:
        payload = [
            {
                "name": output.name,
                "call_id": output.fcall_id,  # NB: this is *function* call id (str)
                "content": output.content,
            }
            for output in outputs
        ]
        try:
            return json.dumps(payload, separators=(",", ":"))
        except TypeError as exc:
            raise ValueError(
                "ask_functions(cache=True) requires JSON-serializable outputs. "
                "Set convert_to_str=True or disable caching."
            ) from exc

    def _deserialize_function_call_outputs(self, payload: str) -> List[FunctionCallOutput]:
        if not payload:
            return []

        try:
            raw_outputs = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("Cached ask_functions payload is malformed.") from exc

        if not isinstance(raw_outputs, list):
            raise ValueError("Cached ask_functions payload must be a list.")

        outputs = []
        for item in raw_outputs:
            if not isinstance(item, dict):
                raise ValueError("Cached ask_functions payload item must be a mapping.")
            outputs.append(
                FunctionCallOutput(
                    content=item.get("content"),
                    name=item.get("name", ""),
                    fcall_id=item.get("call_id", ""),
                )
            )
        return outputs

    def _inject_subagent_context(
        self,
        callme: Callable,
        call_args: dict,
        subagent_name_iter: Optional[Iterator[str]],
    ) -> dict:
        """Allows subagents to be passed into ask_functions as if they were functions."""
        injected_param_names = []
        signature = inspect.signature(callme)
        for param_name, param in signature.parameters.items():
            if _is_agent_context_annotation(param.annotation):
                injected_param_names.append(param_name)

        if not injected_param_names:
            return call_args

        if subagent_name_iter is None:
            raise ValueError(
                "Function call requires AgentContext injection. "
                "Pass subagent_names to ask_functions()."
            )

        try:
            subagent_name = next(subagent_name_iter)
        except StopIteration as exc:
            raise ValueError(
                "Not enough subagent_names provided for AgentContext injection."
            ) from exc

        injected_agent = AgentContext(
            str(subagent_name),
            self._orch,
            ask_params=self.ask_params,
            ignore_cache=self.ignore_cache,
            error_mode=self._error_mode,
        )
        for param_name in injected_param_names:
            call_args[param_name] = injected_agent

        return call_args

    def ask_human(
        self,
        prompt: str,
        documents: Union[
            LLMDocument,
            LLMResponse,
            List[Union[LLMDocument, LLMResponse]],
            MessageState,
            None,
        ],
        *additional_documents: LLMDocument,
        salt: Optional[str] = None,
        input_fn: Optional[Callable[[str], str]] = None,
    ) -> HumanResponse:
        """
        Ask a human for input.

        :param prompt: Prompt shown to the human. Used as instructions/system prompt in hashing.
        :param documents: Documents used as hash basis (analogous to ask_llm input).
        :param additional_documents: Additional documents appended to ``documents``.
        :param salt: Optional salt to differentiate repeated prompts.
        :param input_fn: Optional callable used to collect human input.
            Defaults to built-in ``input``.
        :returns: Human response object.
        """
        seq_id = self._orch.next_seq_id(self.agent_name)

        if isinstance(documents, MessageState):
            documents = list(documents)

        if documents is None:
            doc_list = list(additional_documents)
        else:
            doc_list = reduce_to_list(documents, list(additional_documents))
        resolved_docs = cast_documents(doc_list)

        hashed = compute_hash(prompt, resolved_docs, salt=salt)
        call_id: CallIdentifier = {
            "agent_name": self.agent_name,
            "doc_hash": hashed,
            "seq_id": seq_id,
            "session_id": self._orch.get_session_counter(),
            "meta": {
                "provider_type": None,
                "tag": None,
            },
        }

        datastore = self._orch._backend._get_datastore()
        cached = datastore.retrieve(call_id, origin_type=1)
        if cached is not None:
            if cached.old_session_id is not None:
                call_id["session_id"] = cached.old_session_id
                call_id["seq_id"] = cached.old_seq_id
            return HumanResponse(cached.text, call_id=call_id)

        asker = input_fn or input
        answer = asker(prompt)

        parsed = ParsedResponse(
            text=answer,
            response_id=None,
            metadata=None,
            function_calls=None,
        )
        datastore.store(
            call_id,
            parsed,
            origin_type=1,
        )
        return HumanResponse(answer, call_id=call_id)

    def get_msg_state(self) -> MessageState:
        """
        Get the current MessageState for this agent.

        :returns: The current message state.
        """
        if self._msg_state is None:
            msg_state = MessageState(
                agent_name=self.agent_name,
                true_agent=self,
                ask_params=self.ask_params,
            )
            msg_state = hydrate_msg_state(msg_state, self._orch._backend)
            self._msg_state = msg_state
        return self._msg_state

    def memoize(self, salt=None) -> MemoizeContext:
        """
        Context manager for memoizing.
        When entered, it enables memoization for a **block of logic** the duration of the context.
        This is helpful for expensive or non-deterministic operations:
        this context block will only execute once, and on subsequent runs,
        the results will be replayed from the first execution.
        But note: ONLY changes to MessageState and NonMessageState will be recorded.
        You will not be able to access local variables in this block the 2nd time around.


        :param salt: Optional salt value to differentiate memoization contexts.
            Different salt values will create separate memoization caches.
        """
        return MemoizeContext(self, salt=salt)

    def resolve_all(self, responses: List[LLMResponse]) -> List[str]:
        """
        Resolve all responses at once.

        :param responses: List of LLMResponse objects to resolve.
        :returns: List of resolved string values corresponding to each response.
        """
        return [resp.final_answer for resp in responses]
