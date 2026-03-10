"""Data structures for tracking and replaying MessageState operations."""

from typing import TYPE_CHECKING, Any, List, Union

from pipelinellm.types import LLMDocument, LLMResponse

if TYPE_CHECKING:
    from pipelinellm.core.state.msg_state import MessageState


class Operation:
    """Base class for operations on MessageState."""

    op_type: str


class AppendOp(Operation):
    """Append operation."""

    def __init__(self, item: Union[LLMDocument, LLMResponse]):
        self.op_type = "append"
        self.item = item


class ExtendOp(Operation):
    """Extend operation."""

    def __init__(self, items: List[Union[LLMDocument, LLMResponse]]):
        self.op_type = "extend"
        self.items = items


class InsertOp(Operation):
    """Insert operation."""

    def __init__(self, index: int, item: Union[LLMDocument, LLMResponse]):
        self.op_type = "insert"
        self.index = index
        self.item = item


class SetItemOp(Operation):
    """SetItem operation."""

    def __init__(self, index: int, item: Union[LLMDocument, LLMResponse]):
        self.op_type = "setitem"
        self.index = index
        self.item = item


class PopOp(Operation):
    """Pop operation."""

    def __init__(self, index: int = -1):
        self.op_type = "pop"
        self.index = index


class RemoveOp(Operation):
    """Remove operation."""

    def __init__(self, item: Union[LLMDocument, LLMResponse]):
        self.op_type = "remove"
        self.item = item


class ClearOp(Operation):
    """Clear operation."""

    def __init__(self):
        self.op_type = "clear"


class ReverseOp(Operation):
    """Reverse operation."""

    def __init__(self):
        self.op_type = "reverse"


class SortOp(Operation):
    """Sort operation."""

    def __init__(self, key: Any = None, reverse: bool = False):
        self.op_type = "sort"
        self.key = key
        self.reverse = reverse


class OperationLog:
    """Track all operations performed on a MessageState."""

    def __init__(self):
        self.operations: List[Operation] = []

    def record(self, operation: Operation):
        """Record an operation."""
        self.operations.append(operation)

    def _prepare_for_serialization(self):
        """Prepare operation log for serialization by resolving all lazy values.

        This ensures that LLMResponse objects have their values resolved
        before pickling, so they can be properly restored during replay.
        """

        for op in self.operations:
            if op.op_type == "append":
                if isinstance(op.item, LLMResponse):
                    # Resolve and create a simple LLMResponse with the value
                    resolved_value = op.item.resolve()
                    op.item = LLMResponse(value=resolved_value, call_id=op.item.call_id)
            elif op.op_type == "extend":
                resolved_items = []
                for item in op.items:
                    if isinstance(item, LLMResponse):
                        resolved_value = item.resolve()
                        resolved_items.append(
                            LLMResponse(value=resolved_value, call_id=item.call_id)
                        )
                    else:
                        resolved_items.append(item)
                op.items = resolved_items
            elif op.op_type == "insert":
                if isinstance(op.item, LLMResponse):
                    resolved_value = op.item.resolve()
                    op.item = LLMResponse(value=resolved_value, call_id=op.item.call_id)
            elif op.op_type == "setitem":
                if isinstance(op.item, LLMResponse):
                    resolved_value = op.item.resolve()
                    op.item = LLMResponse(value=resolved_value, call_id=op.item.call_id)

    def replay(self, msg_state: "MessageState"):
        """Replay all recorded operations on a MessageState without tracking.

        :param msg_state: The MessageState to replay operations on.
        """
        # Temporarily disable tracking to avoid recursion
        prev_tracking = getattr(msg_state, "_tracking_operations", False)
        msg_state._tracking_operations = False

        try:
            for op in self.operations:
                if op.op_type == "append":
                    msg_state.data.append(op.item)
                    msg_state._update_seq_counters(op.item)
                elif op.op_type == "extend":
                    msg_state.data.extend(op.items)
                    for item in op.items:
                        msg_state._update_seq_counters(item)
                elif op.op_type == "insert":
                    msg_state.data.insert(op.index, op.item)
                    msg_state._update_seq_counters(op.item)
                elif op.op_type == "setitem":
                    msg_state.data[op.index] = op.item
                    msg_state._update_seq_counters(op.item)
                elif op.op_type == "pop":
                    msg_state.data.pop(op.index)
                elif op.op_type == "remove":
                    msg_state.data.remove(op.item)
                elif op.op_type == "clear":
                    msg_state.data.clear()
                elif op.op_type == "reverse":
                    msg_state.data.reverse()
                elif op.op_type == "sort":
                    msg_state.data.sort(key=op.key, reverse=op.reverse)
        finally:
            msg_state._tracking_operations = prev_tracking

    def clear(self):
        """Clear all recorded operations."""
        self.operations.clear()

    def __len__(self):
        return len(self.operations)
