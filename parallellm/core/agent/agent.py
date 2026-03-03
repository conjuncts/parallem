from typing import TYPE_CHECKING, List, Optional, Union
from parallellm.core.ask import Askable
from parallellm.core.cast.fix_docs import cast_documents, reduce_to_list
from parallellm.core.exception import NotAvailable
from parallellm.core.hash import compute_hash
from parallellm.core.state.msg_state import MessageState
from parallellm.core.response import (
    ReadyLLMResponse,
)
from parallellm.logging.dash_logger import HashStatus
from parallellm.types import (
    AskParameters,
    CallIdentifier,
    CommonQueryParameters,
    HashByOptions,
    LLMDocument,
    LLMIdentity,
    LLMResponse,
    ServerTool,
)

if TYPE_CHECKING:
    from parallellm.core.agent.orchestrator import AgentOrchestrator


class AgentContext(Askable):
    """Context manager for one "agent", which in parallellm is one single autonomous process.

    Parallellm does things a bit differently.
    While typically an agent is associated with a single LLM,
    Parallellm identifies an agent with a process, program, or algorithm
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
        self._persist_msg_state: bool = True

    def __enter__(self):
        # No setup needed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Save message state
        if self._msg_state is not None and self._persist_msg_state:
            self._try_persist_msg_state(self._msg_state)

        if exc_type in (NotAvailable,):
            return True
        if self._orch.strategy == "batch" and exc_type in (NotAvailable,):
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
        text_format: Optional[str] = None,
        tools: Optional[list[Union[dict, ServerTool]]] = None,
        tag: Optional[str] = None,
        save_input: Optional[bool] = None,
        **kwargs,
    ) -> LLMResponse:
        # load ask_params defaults
        for k, v in self.ask_params.items():
            if k == "hash_by" and hash_by is None:
                hash_by = v
            elif k == "save_input" and save_input is None:
                save_input = v

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
            salt_terms.append(salt)
        if hash_by is not None:
            for term in hash_by:
                if term == "llm":
                    if llm is not None:
                        salt_terms.append(llm.identity)
                    else:
                        salt_terms.append(self._orch._provider.provider_type)
        hashed = compute_hash(instructions, resolved_docs + salt_terms)

        if save_input:
            msg_hashes = [compute_hash(None, [msg]) for msg in resolved_docs]
            self._orch._backend._get_datastore().store_doc_hash(
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
            "text_format": text_format,
            "tools": tools,
        }

        return self._orch._backend.submit_query(
            self._orch._provider,
            params,
            call_id=call_id,
            **kwargs,
        )

    def get_msg_state(self, continuation=False) -> MessageState:
        """
        Get the current MessageState for this agent.

        :param continuation: Whether the MessageState should be continued upon exit.
            This lets you save and resume conversations.
            If True, the conversation will always resume where it left off.
            If False, the conversation will be fresh every time. Either way,
            responses still get cached in the backend.
        :returns: The current message state.
        """
        if self._msg_state is None:
            self._msg_state = self._orch.get_msg_state(self)
            self._persist_msg_state = continuation

        return self._msg_state

    def _try_persist_msg_state(self, msg_state):
        self._orch.save_msg_state(
            self,
            msg_state,
        )

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
