from typing import TYPE_CHECKING

from pipelinellm.core.exception import PipelinellmSignal


if TYPE_CHECKING:
    from pipelinellm.core.agent.agent import AgentContext


class MemoizedSignal(PipelinellmSignal):
    """Signal used to indicate that a memoized value should be returned."""

    def __init__(self, value_hash: str):
        super().__init__(
            f"Memoized value with hash {value_hash} found. Returning cached response."
        )
        self.value_hash = value_hash


class MemoizeContext:
    """Context manager for memoization. When entered, it enables memoization for the duration of the context."""

    def __init__(self, agent: "AgentContext"):
        self.agent = agent

        self._msg_state_init_hash = None
        self._memoize_enabled_prev = None

    def __enter__(self):
        # Need to remember the state (doc_hash) of MessageState and NonMessageState
        msg_state = self.agent.get_msg_state()
        self._msg_state_init_hash = msg_state.get_state_hash()
        self._memoize_enabled_prev = msg_state._memoize_enabled
        msg_state._memoize_enabled = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        msg_state = self.agent.get_msg_state()
        msg_state._memoize_enabled = self._memoize_enabled_prev
        # swallow MemoizedSignal
        if exc_type is MemoizedSignal:
            return True
        # now record all modifications to MessageState, so they may be replayed
        # TODO:

    def begin(self):
        """Required to start tracking for memoization. Must be called within the context."""
        # If available, then raise a MemoizedSignal to short circuit and return the cached response.
