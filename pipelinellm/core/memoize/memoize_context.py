from typing import TYPE_CHECKING
import pickle

from pipelinellm.core.exception import PipelinellmSignal
from pipelinellm.core.memoize.operations import OperationLog


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

    def __init__(self, agent: "AgentContext", salt=None):
        self.agent = agent
        self.salt = salt

        self._conv_hash = None
        self._memoize_enabled_prev = None
        self._operation_log = None

    def __enter__(self):

        # Get the MessageState and compute its hash
        msg_state = self.agent.get_msg_state()
        self._conv_hash = msg_state.get_state_hash(salt=self.salt)

        self._memoize_enabled_prev = msg_state._memoize_enabled
        msg_state._memoize_enabled = True

        # Create operation log for tracking
        self._operation_log = OperationLog()
        msg_state._operation_log = self._operation_log

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        msg_state = self.agent.get_msg_state()
        msg_state._memoize_enabled = self._memoize_enabled_prev
        msg_state._tracking_operations = False
        msg_state._operation_log = None

        # swallow MemoizedSignal
        if exc_type is MemoizedSignal:
            return True

        # Store the operation log if there were any operations
        if self._operation_log and len(self._operation_log) > 0:
            # Prepare for serialization (resolve lazy values)
            self._operation_log._prepare_for_serialization()

            datastore = self.agent._orch._backend._get_datastore()
            serialized_log = pickle.dumps(self._operation_log)
            datastore.store_memoize(self._conv_hash, serialized_log)

    def begin(self):
        """Required to start tracking for memoization. Must be called within the context."""
        # Check if there's a cached operation log for this state
        datastore = self.agent._orch._backend._get_datastore()
        serialized_log = datastore.retrieve_memoize(self._conv_hash)

        if serialized_log is not None:
            # Found cached operations - replay them
            operation_log = pickle.loads(serialized_log)
            msg_state = self.agent.get_msg_state()
            operation_log.replay(msg_state)

            # Raise signal to short-circuit execution
            raise MemoizedSignal(self._conv_hash)

        # No cached operations found - start tracking
        msg_state = self.agent.get_msg_state()
        msg_state._tracking_operations = True
