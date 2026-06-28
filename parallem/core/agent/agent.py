import inspect
from functools import partial
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)
from parallem.core.ask import Askable, _raise_exception
from parallem.core.convert._asking import convert_options, convert_tools
from parallem.core.convert.fix_docs import cast_documents, reduce_to_list
from parallem.core.exception import NotAvailable, PendingNotAvailable
from parallem.core.hash import compute_hash, compute_salted_hash
from parallem.core.convert.populate import populate_msg_state
from parallem.core.memoize.memoize_context import MemoizeContext
from parallem.core.state.msg_state import MessageState
from parallem.core.response import (
    ReadyLLMResponse,
)
from parallem.logging.dash_logger import HashStatus
from parallem.tools.auto_schema import _is_agent_context_annotation
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


if TYPE_CHECKING:
    from parallem.core.agent.orchestrator import AgentOrchestrator
    from pydantic import BaseModel


class AgentContext(Askable):
    """Context manager for one "agent", which in ParaLLeM is one single autonomous process.

    ParaLLeM identifies an agent with a process, program, or algorithm
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
        subagent_controller: Optional["SubagentController"] = None,
    ):
        self.agent_name = agent_name
        self._orch = orch
        self._subagent_controller = subagent_controller or SubagentController(self)

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

        llm, provider_type, structured_output = convert_options(
            llm,
            structured_output,
            kwargs,
            provider=self._orch._provider,
        )

        tools = convert_tools(tools)

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
        hashed, salt_terms = compute_salted_hash(
            params,
            salt=salt,
            hash_by=hash_by,
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

        # 3. use cache if available
        cached = self._get_cached_response(call_id)
        if cached is not None:
            self._orch._dashlog.update_call(call_id, HashStatus.CACHED)
            return cached

        # 4. save inputs if needed (pass `params` mapping)
        if save_input:
            self._orch._input_storage.store_input(
                call_id,
                params=params,
                hash_by=hash_by,
                salt=salt,
                request_kwargs=kwargs,
            )

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
        default: Optional[Callable] = _raise_exception,
        convert_to_str: bool = True,
        **kwargs,
    ) -> List[FunctionCallOutput]:
        if functions is None:
            functions = {}
        functions.update(kwargs)
        if not functions:
            raise ValueError(
                "No functions provided to ask_functions. Provide functions as a dict or as kwargs."
            )

        # Check if response has function calls. If so, delegate to user-defined functions.
        fcs = response.function_calls
        fc_outs = []
        for fc in fcs:
            callme = functions.get(fc.name)
            if callme is None:
                # Function not found
                if default is None:
                    continue
                else:
                    fc_outs.append(default())
                    continue

            call_args = dict(fc.args)
            if self._subagent_controller.requires_subagent(callme):
                # If a subagent was requested, then create AgentContext for it
                result = self._subagent_controller.run_subagent(fc.name, callme, call_args)
            else:
                # Otherwise, execute the function directly
                result = callme(**call_args)
            if convert_to_str and not isinstance(result, str):
                result = str(result)
            fc_outs.append(FunctionCallOutput(content=result, name=fc.name, fcall_id=fc.fcall_id))

        return fc_outs

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
            msg_state = populate_msg_state(msg_state, self._orch._backend)
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


class SubagentController:
    """
    Handles the creation of subagents for an agent.
    """
    def __init__(self, parent: "AgentContext"):
        self.parent = parent
        self.existing_subagents = 0

    def generate_subagent_name(self, fc_name: str, func: Callable, call_args: dict):
        """
        Default behavior: name subagents sequentially as
        {parent_agent_name}/{fc_name}_0 etc.
        """
        result = self.parent.agent_name + f"/{fc_name}_{self.existing_subagents}"
        self.existing_subagents += 1
        return result

    def run_subagent(self, fc_name: str, func: Callable, call_args: dict):
        """
        Create and run a subagent.
        An AgentContext is automatically created.
        """
    
        # Create subagent name and context
        subagent_name = self.generate_subagent_name(fc_name, func, call_args)
        subagent_context = AgentContext(
            str(subagent_name),
            self.parent._orch,
            ask_params=self.parent.ask_params,
            ignore_cache=self.parent.ignore_cache,
            error_mode=self.parent._error_mode,
        )

        # Inject into call_args
        signature = inspect.signature(func)
        for param_name, param in signature.parameters.items():
            if _is_agent_context_annotation(param.annotation):
                call_args[param_name] = subagent_context
                break
        else:
            raise ValueError(
                f"Function {func.__name__} requires AgentContext injection, "
                "but no parameter is annotated with AgentContext."
            )

        with subagent_context:
            # Execute the subagent function
            result = func(**call_args)
        return result

    def requires_subagent(
        self,
        callme: Callable,
    ):
        """Check if a function requires an AgentContext parameter."""
        signature = inspect.signature(callme)
        needed_agent_contexts = []
        for _param_name, param in signature.parameters.items():
            if _is_agent_context_annotation(param.annotation):
                needed_agent_contexts.append(param.name)
        if len(needed_agent_contexts) == 1:
            return True
        return None
