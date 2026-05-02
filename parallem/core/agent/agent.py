import inspect
from functools import partial
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
    get_args,
    get_origin,
)
from parallem.core.ask import Askable
from parallem.core.cast.fix_docs import cast_documents, reduce_to_list
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
    HashByOptions,
    HumanResponse,
    LLMDocument,
    LLMIdentity,
    LLMResponse,
    ParsedResponse,
    ServerTool,
)

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
    ):
        self.agent_name = agent_name
        self._orch = orch

        self._seq_id_counter = 0

        self.ask_params = ask_params or {}
        if self.ask_params:
            self.ask_llm = self._bind_ask_llm(self.ask_params)
        self.ignore_cache = ignore_cache

        self._msg_state: Optional[MessageState] = None
        "MessageState for this agent. Some pipelines won't use this (so it will be None)."

    def __enter__(self):
        # No setup needed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type in (NotAvailable, PendingNotAvailable):
            # swallow NotAvailable and its subclasses (like PendingNotAvailable)
            return True
        if self._orch.strategy == "batch" and exc_type in (
            NotAvailable,
            PendingNotAvailable,
        ):
            # swallow NotAvailable errors only in batch mode
            return True
        return False

    def print(self, *args, **kwargs):
        """
        Print to console above the dashboard output.
        This ensures proper display ordering when the dashboard is active.
        """
        self._orch._dashlog.print(*args, **kwargs)

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
                    raise ValueError(
                        f"Tool {tool} is not a dict, ServerTool, or callable."
                    )
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
            llm = LLMIdentity(llm)

        provider_type = self._orch._provider.provider_type
        if provider_type is None:
            provider_type = llm.provider_type

        return llm, provider_type, structured_output

    def _compute_hash(
        self,
        resolved_docs,
        *,
        salt,
        hash_by,
        llm,
        provider_type,
        tools,
        instructions,
    ):
        """Compute the input hash (doc_hash) for a list of documents."""
        # Compute salt
        salt_terms = build_hash_salt_terms(
            salt=salt,
            hash_by=hash_by,
            llm=llm,
            provider_type=provider_type,
            tools=tools,
        )

        # Use a null-byte separator so individual terms cannot be confused with one
        # another, and pass as the `salt` parameter (applied via re-hash) so that
        # salt content can never collide with document content.
        combined_salt = "\x00".join(salt_terms) if salt_terms else None
        hashed = compute_hash(instructions, resolved_docs, salt=combined_salt)
        return hashed, salt_terms

    def _get_cached_response(
        self,
        call_id,
        hashed,
    ):
        """Helper method to check for and return a cached response, if it exists."""
        cached = None if self.ignore_cache else self._orch._backend.retrieve(call_id)
        if cached is not None:
            self.update_hash_status(hashed, HashStatus.CACHED)

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
        salt: Optional[str] = None,
        hash_by: HashByOptions = None,
        structured_output: Optional["BaseModel"] = None,
        tools: Optional[list[Union[dict, ServerTool, Callable]]] = None,
        tag: Optional[str] = None,
        save_input: Optional[bool] = None,
        **kwargs,
    ) -> LLMResponse:
        # 1. assign sequential ID, input checks
        seq_id = self._seq_id_counter
        self._seq_id_counter += 1

        llm, provider_type, structured_output = self._coerce_options(
            llm, structured_output, kwargs
        )
        tools = self._coerce_tools(tools)

        if isinstance(documents, MessageState):
            documents = list(documents)

        documents = reduce_to_list(documents, list(additional_documents))
        resolved_docs = cast_documents(documents)

        # 2. compute hash for inputs
        hashed, salt_terms = self._compute_hash(
            resolved_docs,
            salt=salt,
            hash_by=hash_by,
            llm=llm,
            provider_type=provider_type,
            tools=tools,
            instructions=instructions,
        )

        # 3. save inputs if needed
        if save_input:
            msg_hashes = [compute_hash(None, [msg]) for msg in resolved_docs]
            self._orch._backend._get_datastore().store_input(
                hashed,
                instructions=instructions,
                msgs=documents,
                salt_terms=salt_terms,
                msg_hashes=msg_hashes,
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

        # 4. use cache if available
        cached = self._get_cached_response(call_id, hashed)
        if cached is not None:
            return cached

        # 5. use API
        if not self._orch._provider.is_compatible(provider_type):
            raise ValueError(
                f"LLM {llm.identity} is not compatible with provider {provider_type}"
            )

        params: CommonQueryParameters = {
            "instructions": instructions,
            "strict_documents": resolved_docs,
            "llm": llm,
            "structured_output": structured_output,
            "tools": tools,
        }

        return self._orch._backend.submit_query(
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
        **kwargs,
    ) -> List[FunctionCallOutput]:
        if functions is None:
            functions = {}
        functions.update(kwargs)
        if not functions:
            raise ValueError(
                "No functions provided to ask_functions. Provide functions as a dict or as kwargs."
            )
        subagent_name_iter = (
            iter(subagent_names) if subagent_names is not None else None
        )

        # Check if response has function calls. If so, delegate to user-defined functions.
        fcs = response.function_calls
        fc_outs = []
        for fc in fcs:
            callme = functions.get(fc.name)
            if callme is None:
                # Function not found
                if if_func_not_exist is ValueError:
                    raise ValueError(
                        f"LLM asked for {fc.name}, but it was not provided."
                    )
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
            fc_outs.append(
                FunctionCallOutput(content=result, name=fc.name, call_id=fc.call_id)
            )
        return fc_outs

    def _inject_subagent_context(
        self,
        callme: Callable,
        call_args: dict,
        subagent_name_iter: Optional[Iterator[str]],
    ) -> dict:
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
        seq_id = self._seq_id_counter
        self._seq_id_counter += 1

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

    def update_hash_status(self, hash_value: str, status: HashStatus):
        """
        Update the status of a hash in the logger

        Args:
            hash_value: The hash value to update
            status: New status - one of 'C' (cached), '↗' (sent), '↘' (received), '✓' (stored)
        """
        # only track if asked
        if self._orch._dashlog.display:
            self._orch._dashlog.update_hash(hash_value, status)

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
