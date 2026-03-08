from collections import UserList
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)
from pipelinellm.core.ask import Askable
from pipelinellm.core.cast.fix_docs import reduce_to_list
from pipelinellm.types import (
    FunctionCallOutput,
    HashByOptions,
    LLMDocument,
    LLMIdentity,
    LLMResponse,
    ServerTool,
)

if TYPE_CHECKING:
    from pipelinellm.core.agent.agent import AgentContext


class MessageState(UserList[Union[LLMDocument, LLMResponse]], Askable):
    """
    Simply a list of messages.
    """

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

    def _update_seq_counters(self, other: Union[LLMDocument, LLMResponse]):
        """Update sequence counters based on the other message."""
        if isinstance(other, LLMResponse):
            # recover seq_id
            if other.call_id:
                seq_id = other.call_id.get("seq_id", 0)
                self.anon_ctr = max(self.anon_ctr, seq_id)

    def __setitem__(self, i, item):
        self._update_seq_counters(item)
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
        self.data.append(item)

    def insert(self, i, item):
        self._update_seq_counters(item)
        self.data.insert(i, item)

    def extend(self, others: Iterable[Union[LLMDocument, LLMResponse]], /):
        """Extend this MessageState with a list of other MessageStates."""
        for item in others:
            self._update_seq_counters(item)
        self.data.extend(others)

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
        text_format: Optional[str] = None,
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
            Example: "llm" will also include the LLM name.
        :param text_format: Schema or format specification for structured output.
            For OpenAI: uses structured output via responses.parse().
            For Google: sets response_mime_type and response_schema.
            For Anthropic: not supported.
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
            text_format=text_format,
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

    def persist(self):
        """Persist the current message state to the agent's storage."""
        if self._true_agent:
            self._true_agent._try_persist_msg_state(self)

    def ask_functions(
        self,
        response: Optional[LLMResponse] = None,
        functions: Dict[str, Callable] = None,
        *,
        if_func_not_exist: Union[str, Exception] = ValueError,
        **kwargs,
    ) -> List[FunctionCallOutput]:
        """
        If the agent requested any function calls, then this method actually calls user-defined functions.

        Functions should be provided as kwargs.

        :param functions: Available functions to the model. Mapping from function name to callable.
        :param kwargs: Any additional functions will be added to "functions".
        :param if_func_not_exist: What to do if a function is not found.
            If an Exception is passed, it will be raised. If a string is passed, it will be added to the
            conversation as an error message but allowed to continue.
            Default: ValueError.
        """
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
            last_msg, functions=functions, if_func_not_exist=if_func_not_exist, **kwargs
        )
        self.extend(fc_outs)
        return fc_outs

    def resolve(self) -> List[LLMDocument]:
        """Helper to make sure that all messages have been resolved."""
        resolved = []
        for msg in self.data:
            if isinstance(msg, LLMResponse):
                resolved.append(msg.resolve())
            else:
                resolved.append(msg)
        return resolved
