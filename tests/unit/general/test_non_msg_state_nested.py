"""Unit tests for nested dict/list support in NonMessageState."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from pipelinellm.core.memoize.operations import (
    # AppendNestedListOp,
    # ClearNestedDictOp,
    # ClearNestedListOp,
    # DelNestedDictItemOp,
    # DelNestedListItemOp,
    # ExtendNestedListOp,
    # InsertNestedListOp,
    OperationLog,
    # PopNestedListOp,
    # RemoveNestedListOp,
    # ReverseNestedListOp,
    # SetNestedDictItemOp,
    # SetNestedListItemOp,
    # SortNestedListOp,
)
from pipelinellm.core.state.non_msg_state import (
    NonMessageState,
    # TrackedDict,
    # TrackedList,
)

if TYPE_CHECKING:
    from pipelinellm.core.memoize.operations import (
        AppendNestedListOp,
        ClearNestedDictOp,
        ClearNestedListOp,
        DelNestedDictItemOp,
        DelNestedListItemOp,
        ExtendNestedListOp,
        InsertNestedListOp,
        PopNestedListOp,
        RemoveNestedListOp,
        ReverseNestedListOp,
        SetNestedDictItemOp,
        SetNestedListItemOp,
        SortNestedListOp,
    )
    from pipelinellm.core.state.non_msg_state import (
        TrackedDict,
        TrackedList,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state() -> NonMessageState:
    backend = MagicMock()
    return NonMessageState(backend)


def _make_tracking_state() -> tuple[NonMessageState, OperationLog]:
    state = _make_state()
    log = OperationLog()
    state._tracking_operations = True
    state._operation_log = log
    return state, log


# ---------------------------------------------------------------------------
# TrackedDict – basic mutation tracking
# ---------------------------------------------------------------------------


@pytest.mark.skip("Not implemented")
class TestTrackedDictTracking:
    def test_setitem_records_op(self):
        state, log = _make_tracking_state()
        state["d"] = {}
        d = state["d"]
        d["x"] = 1
        # Two ops: SetNonMsgItemOp + SetNestedDictItemOp
        assert len(log) == 2
        op = log.operations[1]
        assert isinstance(op, SetNestedDictItemOp)
        assert op.key == "d"
        assert op.nested_key == "x"
        assert op.value == 1

    def test_delitem_records_op(self):
        state, log = _make_tracking_state()
        state["d"] = {"a": 1}
        d = state["d"]
        del d["a"]
        assert isinstance(log.operations[-1], DelNestedDictItemOp)
        assert log.operations[-1].nested_key == "a"

    def test_update_records_ops(self):
        state, log = _make_tracking_state()
        state["d"] = {}
        d = state["d"]
        d.update({"p": 1, "q": 2})
        set_ops = [o for o in log.operations if isinstance(o, SetNestedDictItemOp)]
        assert len(set_ops) == 2

    def test_pop_records_del_op(self):
        state, log = _make_tracking_state()
        state["d"] = {"k": 99}
        d = state["d"]
        val = d.pop("k")
        assert val == 99
        assert isinstance(log.operations[-1], DelNestedDictItemOp)

    def test_pop_missing_key_with_default_no_op(self):
        state, log = _make_tracking_state()
        state["d"] = {}
        d = state["d"]
        ops_before = len(log)
        d.pop("missing", None)
        assert len(log) == ops_before

    def test_clear_records_op(self):
        state, log = _make_tracking_state()
        state["d"] = {"a": 1}
        d = state["d"]
        d.clear()
        assert isinstance(log.operations[-1], ClearNestedDictOp)
        assert log.operations[-1].key == "d"

    def test_setdefault_sets_and_records(self):
        state, log = _make_tracking_state()
        state["d"] = {}
        d = state["d"]
        val = d.setdefault("k", 7)
        assert val == 7
        assert isinstance(log.operations[-1], SetNestedDictItemOp)

    def test_setdefault_existing_no_op(self):
        state, log = _make_tracking_state()
        state["d"] = {"k": 5}
        d = state["d"]
        ops_before = len(log)
        d.setdefault("k", 99)
        assert len(log) == ops_before

    def test_no_tracking_when_disabled(self):
        state = _make_state()
        state["d"] = {}
        d = state["d"]
        d["x"] = 1
        # No log attached – should not raise
        assert d["x"] == 1


# ---------------------------------------------------------------------------
# TrackedList – basic mutation tracking
# ---------------------------------------------------------------------------


@pytest.mark.skip("Not implemented")
class TestTrackedListTracking:
    def test_append_records_op(self):
        state, log = _make_tracking_state()
        state["lst"] = []
        lst = state["lst"]
        lst.append(42)
        assert isinstance(log.operations[-1], AppendNestedListOp)
        assert log.operations[-1].value == 42

    def test_extend_records_op(self):
        state, log = _make_tracking_state()
        state["lst"] = []
        lst = state["lst"]
        lst.extend([1, 2, 3])
        op = log.operations[-1]
        assert isinstance(op, ExtendNestedListOp)
        assert op.items == [1, 2, 3]

    def test_insert_records_op(self):
        state, log = _make_tracking_state()
        state["lst"] = [1, 2]
        lst = state["lst"]
        lst.insert(1, 99)
        op = log.operations[-1]
        assert isinstance(op, InsertNestedListOp)
        assert op.index == 1
        assert op.value == 99

    def test_setitem_records_op(self):
        state, log = _make_tracking_state()
        state["lst"] = [0, 1, 2]
        lst = state["lst"]
        lst[1] = 77
        op = log.operations[-1]
        assert isinstance(op, SetNestedListItemOp)
        assert op.index == 1
        assert op.value == 77

    def test_delitem_records_op(self):
        state, log = _make_tracking_state()
        state["lst"] = [10, 20, 30]
        lst = state["lst"]
        del lst[0]
        op = log.operations[-1]
        assert isinstance(op, DelNestedListItemOp)
        assert op.index == 0

    def test_pop_records_op_with_resolved_index(self):
        state, log = _make_tracking_state()
        state["lst"] = [1, 2, 3]
        lst = state["lst"]
        val = lst.pop()  # pops index -1 → actual 2
        assert val == 3
        op = log.operations[-1]
        assert isinstance(op, PopNestedListOp)
        assert op.index == 2

    def test_remove_records_op_by_index(self):
        state, log = _make_tracking_state()
        state["lst"] = ["a", "b", "c"]
        lst = state["lst"]
        lst.remove("b")
        op = log.operations[-1]
        assert isinstance(op, RemoveNestedListOp)
        assert op.index == 1

    def test_clear_records_op(self):
        state, log = _make_tracking_state()
        state["lst"] = [1, 2]
        lst = state["lst"]
        lst.clear()
        assert isinstance(log.operations[-1], ClearNestedListOp)

    def test_reverse_records_op(self):
        state, log = _make_tracking_state()
        state["lst"] = [1, 2, 3]
        lst = state["lst"]
        lst.reverse()
        assert isinstance(log.operations[-1], ReverseNestedListOp)

    def test_sort_records_op(self):
        state, log = _make_tracking_state()
        state["lst"] = [3, 1, 2]
        lst = state["lst"]
        lst.sort()
        op = log.operations[-1]
        assert isinstance(op, SortNestedListOp)
        assert op.reverse is False

    def test_sort_reverse_records_op(self):
        state, log = _make_tracking_state()
        state["lst"] = [1, 2, 3]
        lst = state["lst"]
        lst.sort(reverse=True)
        op = log.operations[-1]
        assert isinstance(op, SortNestedListOp)
        assert op.reverse is True

    def test_sort_with_key_raises(self):
        state = _make_state()
        state["lst"] = [1, 2, 3]
        lst = state["lst"]
        with pytest.raises(TypeError):
            lst.sort(key=lambda x: x)


# ---------------------------------------------------------------------------
# NonMessageState – storage and retrieval
# ---------------------------------------------------------------------------


@pytest.mark.skip("Not implemented")
class TestNonMessageStateContainers:
    def test_dict_stored_as_tracked(self):
        state = _make_state()
        state["d"] = {"a": 1}
        assert isinstance(state["d"], TrackedDict)

    def test_list_stored_as_tracked(self):
        state = _make_state()
        state["lst"] = [1, 2]
        assert isinstance(state["lst"], TrackedList)

    def test_plain_dict_wrapped_lazily_on_getitem(self):
        state = _make_state()
        state.data["d"] = {"x": 1}  # bypass __setitem__
        assert isinstance(state["d"], TrackedDict)

    def test_plain_list_wrapped_lazily_on_getitem(self):
        state = _make_state()
        state.data["lst"] = [1, 2]
        assert isinstance(state["lst"], TrackedList)

    def test_invalid_type_raises(self):
        state = _make_state()
        with pytest.raises(TypeError):
            state["bad"] = 123

    def test_get_returns_tracked_dict(self):
        state = _make_state()
        state["d"] = {}
        assert isinstance(state.get("d"), TrackedDict)

    def test_get_missing_returns_default(self):
        state = _make_state()
        assert state.get("missing", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# Replay – OperationLog.replay reconstructs nested state
# ---------------------------------------------------------------------------


@pytest.mark.skip("Not implemented")
class TestNestedReplay:
    def _replay_on_fresh(self, log: OperationLog) -> NonMessageState:
        fresh = _make_state()
        from pipelinellm.core.state.msg_state import MessageState

        msg = MessageState()
        log.replay(msg, fresh)
        return fresh

    def test_dict_set_and_del_replay(self):
        state, log = _make_tracking_state()
        state["d"] = {"a": 1, "b": 2}
        d = state["d"]
        d["c"] = 3
        del d["a"]
        fresh = self._replay_on_fresh(log)
        result = fresh["d"]
        assert result == {"b": 2, "c": 3}

    def test_dict_clear_replay(self):
        state, log = _make_tracking_state()
        state["d"] = {"a": 1}
        state["d"].clear()
        fresh = self._replay_on_fresh(log)
        assert fresh["d"] == {}

    def test_list_append_extend_replay(self):
        state, log = _make_tracking_state()
        state["lst"] = []
        lst = state["lst"]
        lst.append(1)
        lst.extend([2, 3])
        fresh = self._replay_on_fresh(log)
        assert list(fresh["lst"]) == [1, 2, 3]

    def test_list_insert_setitem_replay(self):
        state, log = _make_tracking_state()
        state["lst"] = [1, 3]
        lst = state["lst"]
        lst.insert(1, 2)
        lst[0] = 10
        fresh = self._replay_on_fresh(log)
        assert list(fresh["lst"]) == [10, 2, 3]

    def test_list_pop_replay(self):
        state, log = _make_tracking_state()
        state["lst"] = [1, 2, 3]
        state["lst"].pop()
        fresh = self._replay_on_fresh(log)
        assert list(fresh["lst"]) == [1, 2]

    def test_list_remove_replay(self):
        state, log = _make_tracking_state()
        state["lst"] = ["a", "b", "c"]
        state["lst"].remove("b")
        fresh = self._replay_on_fresh(log)
        assert list(fresh["lst"]) == ["a", "c"]

    def test_list_reverse_replay(self):
        state, log = _make_tracking_state()
        state["lst"] = [1, 2, 3]
        state["lst"].reverse()
        fresh = self._replay_on_fresh(log)
        assert list(fresh["lst"]) == [3, 2, 1]

    def test_list_sort_replay(self):
        state, log = _make_tracking_state()
        state["lst"] = [3, 1, 2]
        state["lst"].sort(reverse=True)
        fresh = self._replay_on_fresh(log)
        assert list(fresh["lst"]) == [3, 2, 1]

    def test_list_clear_replay(self):
        state, log = _make_tracking_state()
        state["lst"] = [1, 2, 3]
        state["lst"].clear()
        fresh = self._replay_on_fresh(log)
        assert list(fresh["lst"]) == []

    def test_full_sequence_replay(self):
        state, log = _make_tracking_state()
        state["meta"] = {}
        state["tags"] = []
        state["meta"]["status"] = "running"
        state["tags"].append("alpha")
        state["tags"].append("beta")
        state["meta"]["count"] = 2

        fresh = self._replay_on_fresh(log)
        assert fresh["meta"]["status"] == "running"
        assert fresh["meta"]["count"] == 2
        assert list(fresh["tags"]) == ["alpha", "beta"]
