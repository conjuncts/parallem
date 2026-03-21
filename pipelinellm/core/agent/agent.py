import warnings
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union
from pipelinellm.core.ask import Askable
from pipelinellm.core.cast.fix_docs import cast_documents, reduce_to_list
from pipelinellm.core.exception import NotAvailable, PendingNotAvailable
from pipelinellm.core.hash import compute_hash
from pipelinellm.core.hydrate import hydrate_msg_state
from pipelinellm.core.memoize.memoize_context import MemoizeContext
from pipelinellm.core.state.msg_state import MessageState
from pipelinellm.core.response import (
    ReadyLLMResponse,
)
from pipelinellm.logging.dash_logger import HashStatus
from pipelinellm.types import (
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
    from pipelinellm.core.agent.orchestrator import AgentOrchestrator


class AgentContext(Askable):
    """Context manager for one "agent", which in pipelinellm is one single autonomous process.

    pipelinellm does things a bit differently.
    While typically an agent is associated with a single LLM,
    pipelinellm identifies an agent with a process, program, or algorithm
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

        self._anonymous_counter = 0

        self.ask_params = ask_params or {}
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

    @property
    def my_metadata(self) -> dict:
        """
        Metadata for this agent.
        Backed by the AgentOrchestrator's FileManager.
        """
        key = self.agent_name if self.agent_name is not None else ""
        return self._orch._fm.metadata["agents"].setdefault(
            key,
            {},
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
        structured_output: Optional[object] = None,
        tools: Optional[list[Union[dict, ServerTool]]] = None,
        tag: Optional[str] = None,
        save_input: Optional[bool] = None,
        _legacy_salt: Optional[str] = None,
        _legacy_hash_by: HashByOptions = None,
        **kwargs,
    ) -> LLMResponse:
        # Handle legacy text_format alias
        legacy_text_format = kwargs.pop("text_format", None)
        if structured_output is not None and legacy_text_format is not None:
            raise ValueError(
                "Cannot specify both structured_output and text_format. "
                "text_format is a legacy alias for structured_output."
            )
        if structured_output is None:
            structured_output = legacy_text_format

        # load ask_params defaults
        for k, v in self.ask_params.items():
            if k == "hash_by" and hash_by is None:
                hash_by = v
            elif k == "save_input" and save_input is None:
                save_input = v
            elif k == "llm" and llm is None:
                llm = v
            elif k == "structured_output" and structured_output is None:
                structured_output = v
            elif k == "text_format" and structured_output is None:
                structured_output = v

        if llm is None:
            llm = self._orch._provider.get_default_llm_identity()
        elif isinstance(llm, str):
            llm = LLMIdentity(llm)

        seq_id = self._anonymous_counter
        self._anonymous_counter += 1

        if isinstance(documents, MessageState):
            documents = list(documents)

        documents = reduce_to_list(documents, list(additional_documents))
        resolved_docs = cast_documents(documents)

        # Compute salt
        salt_terms: list[str] = []
        if salt is not None:
            salt_terms.append(str(salt))
        if hash_by is not None:
            for term in hash_by:
                if term == "llm":
                    if llm is not None:
                        salt_terms.append(llm.identity)
                    else:
                        salt_terms.append(self._orch._provider.provider_type)

        use_legacy = _legacy_salt is not None or _legacy_hash_by is not None
        if use_legacy:
            warnings.warn(
                "_legacy_salt and _legacy_hash_by are deprecated and exist only for "
                "migration purposes. Switch to salt/hash_by to avoid hash collisions.",
                DeprecationWarning,
                stacklevel=2,
            )
            legacy_terms: list[str] = []
            if _legacy_salt is not None:
                legacy_terms.append(str(_legacy_salt))
            if _legacy_hash_by is not None:
                for term in _legacy_hash_by:
                    if term == "llm":
                        if llm is not None:
                            legacy_terms.append(llm.identity)
                        else:
                            legacy_terms.append(self._orch._provider.provider_type)
            # Also fold in any normal salt/hash_by terms (unlikely to mix, but safe)
            legacy_terms = salt_terms + legacy_terms
            hashed = compute_hash(
                instructions, resolved_docs, _legacy_salt=legacy_terms
            )
        else:
            # Use a null-byte separator so individual terms cannot be confused with one
            # another, and pass as the `salt` parameter (applied via re-hash) so that
            # salt content can never collide with document content.
            combined_salt = "\x00".join(salt_terms) if salt_terms else None
            hashed = compute_hash(instructions, resolved_docs, salt=combined_salt)

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
                "provider_type": self._orch._provider.provider_type,
                "tag": tag,
            },
        }

        # Cache using datastore
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

        if not self._orch._provider.is_compatible(llm.provider_type):
            raise ValueError(
                f"LLM {llm.identity} is not compatible"
                + f" with provider {self._orch._provider.provider_type}"
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

    def ask_functions(
        self,
        response: LLMResponse,
        functions: Dict[str, Callable] = None,
        *,
        if_func_not_exist: Union[str, Exception] = ValueError,
        **kwargs,
    ) -> List[FunctionCallOutput]:
        if functions is None:
            functions = {}
        functions.update(kwargs)

        # Check if response has function calls. If so, delegate to user-defined functions.
        fcs = response.resolve_function_calls()
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
                else:
                    fc_outs.append(if_func_not_exist)
                    continue

            # Execute the function
            result = callme(**fc.args)
            fc_outs.append(
                FunctionCallOutput(content=result, name=fc.name, call_id=fc.call_id)
            )
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
        Ask a human for input and persist the response.

        Human responses are written to the datastore with ``origin_type=1`` so they
        are never mixed with cached LLM responses.

        :param prompt: Prompt shown to the human. Used as instructions/system prompt in hashing.
        :param documents: Documents used as hash basis (analogous to ask_llm input).
        :param additional_documents: Additional documents appended to ``documents``.
        :param salt: Optional salt to differentiate repeated prompts.
        :param input_fn: Optional callable used to collect human input.
            Defaults to built-in ``input``.
        :returns: Human response object.
        """
        seq_id = self._anonymous_counter
        self._anonymous_counter += 1

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
            msg_state = MessageState(agent_name=self.agent_name, true_agent=self)
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
        return [resp.resolve() for resp in responses]
