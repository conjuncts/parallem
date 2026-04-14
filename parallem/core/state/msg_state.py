from collections import UserList
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)
from parallem.core.ask import Askable
from parallem.core.cast.fix_docs import reduce_to_list
from parallem.core.hash import compute_hash
from parallem.core.memoize.operations import (
    AppendOp,
    ClearOp,
    ExtendOp,
    InsertOp,
    OperationLog,
    PopOp,
    RemoveOp,
    ReverseOp,
    SetItemOp,
    SortOp,
)
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
    from parallem.core.agent.agent import AgentContext
    from parallem.core.memoize.operations import OperationLog
    from pydantic import BaseModel


class MessageState(UserList[Union[LLMDocument, LLMResponse]], Askable):
    """
    Simply a list of messages.
    """

    _CONTINUED_STATE_HASH = "__continued_msg_state__"  # safe, since not 64 chars

    def __init__(
        self,
        initlist=None,
        *,
        agent_name: str = None,
        anon_ctr=0,
        chkp_ctr=0,
        true_agent: "AgentContext" = None,
    ):
        super().__init__(initlist)
        self.agent_name = agent_name
        self.anon_ctr = anon_ctr
        self.chkp_ctr = chkp_ctr
        self._true_agent = true_agent
        self._memoize_enabled = False
        self._tracking_operations = False
        self._operation_log: Optional["OperationLog"] = None
        self._continued_operation_log: Optional["OperationLog"] = None
        self._continued_mode = False

    def copy(self) -> "MessageState":
        """Create a copy of this MessageState."""
        new_state = MessageState(
            agent_name=self.agent_name,
            anon_ctr=self.anon_ctr,
            chkp_ctr=self.chkp_ctr,
            true_agent=self._true_agent,
        )
        new_state.data = self.data.copy()
        return new_state

    def get_state_hash(self, salt=None) -> str:
        """Compute a hash of the current MessageState.

        :return: Hash string representing the current state.
        """
        # Hash based on the contents of the message list and counters
        docs = self.data
        if salt is not None:
            docs = docs
        return compute_hash(None, docs, salt=salt)

    def _track_operation(self, operation):
        """Track an operation if operation tracking is enabled.

        :param operation: The operation to track.
        """
        if self._tracking_operations and self._operation_log is not None:
            self._operation_log.record(operation)

    def _update_seq_counters(self, other: Union[LLMDocument, LLMResponse]):
        """Update sequence counters based on the other message."""
        if isinstance(other, LLMResponse):
            # recover seq_id
            if other.call_id:
                seq_id = other.call_id.get("seq_id", 0)
                self.anon_ctr = max(self.anon_ctr, seq_id)

    def __setitem__(self, i, item):
        self._update_seq_counters(item)
        self._track_operation(SetItemOp(i, item))
        super().__setitem__(i, item)

    def __add__(self, other):
        other_list = []
        if isinstance(other, UserList):
            other_list = other.data
        elif isinstance(other, type(self.data)):
            other_list = other
        else:
            other_list = list(other)

        ret = self.__class__(self.data + other_list)
        for item in other_list:
            ret._update_seq_counters(item)
        return ret

    def __radd__(self, other):
        other_list = []
        if isinstance(other, UserList):
            other_list = other.data
        elif isinstance(other, type(self.data)):
            other_list = other
        else:
            other_list = list(other)
        ret = self.__class__(other_list + self.data)
        for item in other_list:
            ret._update_seq_counters(item)
        return ret

    def __iadd__(self, other):
        other_list = []
        if isinstance(other, UserList):
            other_list = other.data
        elif isinstance(other, type(self.data)):
            other_list = other
        else:
            other_list = list(other)
        self.data += other_list
        for item in other_list:
            self._update_seq_counters(item)
        return self

    def append(self, item: Union[LLMDocument, LLMResponse], /):
        """Append another MessageState to this one and return a new MessageState."""
        self._update_seq_counters(item)
        self._track_operation(AppendOp(item))
        self.data.append(item)

    def insert(self, i, item):
        self._update_seq_counters(item)
        self._track_operation(InsertOp(i, item))
        self.data.insert(i, item)

    def extend(self, others: Iterable[Union[LLMDocument, LLMResponse]], /):
        """Extend this MessageState with a list of other MessageStates."""
        others_list = list(others)
        for item in others_list:
            self._update_seq_counters(item)
        self._track_operation(ExtendOp(others_list))
        self.data.extend(others_list)

    def pop(self, i: int = -1):
        """Remove and return item at index (default last)."""
        self._track_operation(PopOp(i))
        return self.data.pop(i)

    def remove(self, item: Union[LLMDocument, LLMResponse]):
        """Remove first occurrence of item."""
        self._track_operation(RemoveOp(item))
        self.data.remove(item)

    def clear(self):
        """Remove all items from list."""
        self._track_operation(ClearOp())
        self.data.clear()

    def reverse(self):
        """Reverse list in place."""
        self._track_operation(ReverseOp())
        self.data.reverse()

    def sort(self, *, key=None, reverse=False):
        """Sort list in place."""
        if key is not None:
            raise ValueError(
                "MessageState.sort(key=...) is not supported because key functions are not serializable safely"
            )
        self._track_operation(SortOp(reverse))
        self.data.sort(key=key, reverse=reverse)

    def ask_llm(
        self,
        documents: Union[
            LLMDocument,
            LLMResponse,
            List[Union[LLMDocument, LLMResponse]],
            "MessageState",
        ] = None,
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
        Ask the LLM a question. By asking a question directly on the MessageState,
        new documents and the response
        automatically gets appended to the conversation.

        :param documents: Documents to use, such as the prompt.
            Can be strings or images.
        :param instructions: The system prompt to use.
        :param llm: The identity of the LLM to use.
            Can be helpful multi-agent or multi-model scenarios.
        :param salt: A value to include in the hash for differentiation.
        :param hash_by: The names of additional terms to include in the hash for differentiation.
            Example: "llm" includes the LLM name.
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
        if documents is not None:
            self.extend(reduce_to_list(documents))
            self.extend(list(additional_documents))
        out = self._true_agent.ask_llm(
            self,
            instructions=instructions,
            llm=llm,
            salt=salt,
            hash_by=hash_by,
            structured_output=structured_output,
            tools=tools,
            tag=tag,
            save_input=save_input,
            **kwargs,
        )
        self._update_seq_counters(out)
        self.append(out)
        return out

    def __getstate__(self):
        # Exclude _true_agent from pickling
        state = self.__dict__.copy()
        del state["_true_agent"]
        return state

    def save(self):
        """Persist the current message state to the agent's storage."""
        if self._true_agent is None:
            return

        datastore = self._true_agent._orch._backend._get_datastore()
        snapshot_log = OperationLog()
        snapshot_log.record(ClearOp())
        snapshot_log.record(ExtendOp(list(self.data)))
        datastore.store_memoize(
            self._true_agent.agent_name,
            self._CONTINUED_STATE_HASH,
            snapshot_log,
        )

    def load(self) -> "MessageState":
        """Enable continued mode for this MessageState.

        Continued mode automatically replays previously recorded operations,
        and starts recording new operations.

        Saving is explicit: call ``persist()`` when you want to store
        accumulated operations.

        :returns: The same MessageState instance in continued mode.
        """

        if self._true_agent is not None:
            # Can proceed with loading state
            self.clear()

            datastore = self._true_agent._orch._backend._get_datastore()
            oplog = datastore.retrieve_memoize(
                self._true_agent.agent_name,
                self._CONTINUED_STATE_HASH,
            )
            if oplog is not None:
                oplog.replay(self)

        return self

    def ask_functions(
        self,
        response: Optional[LLMResponse] = None,
        functions: Dict[str, Callable] = None,
        *,
        subagent_names: Optional[Sequence[str]] = None,
        if_func_not_exist: Union[str, Exception] = None,
        **kwargs,
    ) -> List[FunctionCallOutput]:
        if response is None:
            # Obtain last message from LLM; check if it made any function calls
            if len(self) <= 0:
                # Nothing to do??
                return
            last_msg = self[-1]
            if not isinstance(last_msg, LLMResponse):
                # Nothing to do
                return
        else:
            last_msg = response
        fc_outs = self._true_agent.ask_functions(
            last_msg,
            functions=functions,
            subagent_names=subagent_names,
            if_func_not_exist=if_func_not_exist,
            **kwargs,
        )
        self.extend(fc_outs)
        return fc_outs

    def ask_human(
        self,
        prompt: str,
        documents: Union[
            LLMDocument,
            LLMResponse,
            List[Union[LLMDocument, LLMResponse]],
            "MessageState",
            None,
        ] = None,
        *additional_documents: LLMDocument,
        salt: Optional[str] = None,
        input_fn: Optional[Callable[[str], str]] = None,
    ) -> HumanResponse:
        """
        Ask a human for input and append that response to this message state.

        :param prompt: Prompt shown to the human.
        :param documents: Optional documents to append before asking.
        :param additional_documents: Additional documents to append before asking.
        :param salt: Optional salt to differentiate repeated prompts.
        :param input_fn: Optional callable used to collect human input.
            Defaults to built-in ``input``.
        :returns: Human response object.
        """
        if self._true_agent is None:
            raise ValueError("MessageState is not attached to an agent")

        if documents is not None:
            self.extend(reduce_to_list(documents))
            self.extend(list(additional_documents))
        elif additional_documents:
            self.extend(list(additional_documents))

        out = self._true_agent.ask_human(
            prompt,
            self,
            salt=salt,
            input_fn=input_fn,
        )
        self.append(out)
        return out

    def resolve(self) -> List[LLMDocument]:
        """Helper to make sure that all messages have been resolved."""
        resolved = []
        for msg in self.data:
            if isinstance(msg, LLMResponse):
                resolved.append(msg.resolve())
            else:
                resolved.append(msg)
        return resolved
