"""Data structures for tracking and replaying MessageState operations."""

from typing import TYPE_CHECKING, Any, List, Optional, Union

from parallem.types import LLMDocument, LLMResponse

if TYPE_CHECKING:
    from parallem.core.state.msg_state import MessageState
    from parallem.core.state.non_msg_state import NonMessageState


OperationItem = Union[LLMDocument, LLMResponse]


class Operation:
    """Base class for operations on MessageState or NonMessageState."""

    op_type: str
    item: Optional[OperationItem]
    items: List[OperationItem]
    index: Optional[int]
    reverse: bool
    key: Optional[str]
    value: Optional[Any]

    def __init__(
        self,
        op_type: str,
        *,
        item: Optional[OperationItem] = None,
        items: Optional[List[OperationItem]] = None,
        index: Optional[int] = None,
        reverse: bool = False,
        key: Optional[str] = None,
        value: Optional[Any] = None,
    ):
        self.op_type = op_type
        self.item = item
        self.items = list(items) if items is not None else []
        self.index = index
        self.reverse = reverse
        self.key = key
        self.value = value


class AppendOp(Operation):
    """Append operation."""

    def __init__(self, item: OperationItem):
        super().__init__("append", item=item)


class ExtendOp(Operation):
    """Extend operation."""

    def __init__(self, items: List[OperationItem]):
        super().__init__("extend", items=items)


class InsertOp(Operation):
    """Insert operation."""

    def __init__(self, index: int, item: OperationItem):
        super().__init__("insert", index=index, item=item)


class SetItemOp(Operation):
    """SetItem operation."""

    def __init__(self, index: int, item: OperationItem):
        super().__init__("setitem", index=index, item=item)


class PopOp(Operation):
    """Pop operation."""

    def __init__(self, index: int = -1):
        super().__init__("pop", index=index)


class RemoveOp(Operation):
    """Remove operation."""

    def __init__(self, item: OperationItem):
        super().__init__("remove", item=item)


class ClearOp(Operation):
    """Clear operation."""

    def __init__(self):
        super().__init__("clear")


class ReverseOp(Operation):
    """Reverse operation."""

    def __init__(self):
        super().__init__("reverse")


class SortOp(Operation):
    """Sort operation."""

    def __init__(self, reverse: bool = False):
        super().__init__("sort", reverse=reverse)


class SetNonMsgItemOp(Operation):
    """SetItem operation for NonMessageState."""

    def __init__(self, key: str, value: OperationItem):
        super().__init__("setnonmsgitem", key=key, value=value)


class OperationLog:
    """Track all operations performed on a MessageState."""

    def __init__(self):
        self.operations: List[Operation] = []

    def record(self, operation: Operation):
        """Record an operation."""
        self.operations.append(operation)

    def extend(self, other: "OperationLog"):
        """Extend this log with another OperationLog."""
        self.operations.extend(other.operations)

    def replay(
        self,
        msg_state: "MessageState",
        non_msg_state: "Optional[NonMessageState]" = None,
    ):
        """Replay all recorded operations without tracking.

        :param msg_state: The MessageState to replay operations on.
        :param non_msg_state: Optional NonMessageState for non-message operations.
        """
        # Temporarily disable tracking to avoid recursion
        prev_tracking = getattr(msg_state, "_tracking_operations", False)
        msg_state._tracking_operations = False
        prev_non_msg_tracking = False
        if non_msg_state is not None:
            prev_non_msg_tracking = getattr(
                non_msg_state, "_tracking_operations", False
            )
            non_msg_state._tracking_operations = False

        try:
            for op in self.operations:
                if op.op_type == "setnonmsgitem":
                    if non_msg_state is None:
                        continue
                    non_msg_state.data[op.key] = op.value
                    continue

                item = op.item
                items = op.items
                if op.op_type == "append":
                    msg_state.data.append(item)
                    msg_state._update_seq_counters(item)
                elif op.op_type == "extend":
                    msg_state.data.extend(items)
                    for item in items:
                        msg_state._update_seq_counters(item)
                elif op.op_type == "insert":
                    msg_state.data.insert(op.index, item)
                    msg_state._update_seq_counters(item)
                elif op.op_type == "setitem":
                    msg_state.data[op.index] = item
                    msg_state._update_seq_counters(item)
                elif op.op_type == "pop":
                    msg_state.data.pop(op.index)
                elif op.op_type == "remove":
                    msg_state.data.remove(item)
                elif op.op_type == "clear":
                    msg_state.data.clear()
                elif op.op_type == "reverse":
                    msg_state.data.reverse()
                elif op.op_type == "sort":
                    msg_state.data.sort(reverse=op.reverse)
        finally:
            msg_state._tracking_operations = prev_tracking
            if non_msg_state is not None:
                non_msg_state._tracking_operations = prev_non_msg_tracking

    def clear(self):
        """Clear all recorded operations."""
        self.operations.clear()

    def __len__(self):
        return len(self.operations)
