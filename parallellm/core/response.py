from typing import TYPE_CHECKING
from parallellm.core.calls import _call_to_concise_dict
from parallellm.types import LLMResponse
from parallellm.core.exception import NotAvailable
from parallellm.types import (
    CallIdentifier,
    ParsedResponse,
)

if TYPE_CHECKING:
    from parallellm.core.backend import BaseBackend


class PendingLLMResponse(LLMResponse):
    """
    Base class for a response from an LLM.
    """

    def __init__(
        self,
        call_id: CallIdentifier,
        backend: "BaseBackend",
    ):
        super().__init__(value=None, call_id=call_id)
        self._backend = backend

    def resolve(self) -> str:
        if self.value is not None:
            return self.value

        pr = self._backend.retrieve(self.call_id)
        self.value = pr.text if pr else None
        self._pr = pr
        return self.value

    def __getstate__(self):
        """
        Support for pickling. Only store the call_id since that uniquely identifies the response.
        """
        return {"call_id": _call_to_concise_dict(self.call_id)}

    def __setstate__(self, state):
        """
        Support for unpickling. Restore the call_id, but value will need to be resolved later.
        """
        # self.call_id = _concise_dict_to_call(state["call_id"])
        self.call_id = state["call_id"]
        self.value = None
        self._pr = None
        self._backend = None  # Will be set later


class ReadyLLMResponse(LLMResponse):
    """
    A response that is already resolved.
    """

    def __init__(
        self, call_id: CallIdentifier, *, pr: ParsedResponse = None, value: str = None
    ):
        super().__init__(value=pr.text if pr else value, call_id=call_id)
        self._pr = pr

    def __getstate__(self):
        """
        Support for pickling. Only store the call_id since that uniquely identifies the response.
        """
        return {"call_id": _call_to_concise_dict(self.call_id)}

    def __setstate__(self, state):
        """
        Support for unpickling. Restore the call_id, but value will need to be resolved later.
        """
        self.call_id = state["call_id"]
        self.value = None
        self._pr = None


class BatchLLMResponse(LLMResponse):
    """
    A response that is pending and cannot be resolved due to being
    sent in a batch.
    """

    def __init__(
        self,
        call_id: CallIdentifier,
    ):
        super().__init__(value=None, call_id=call_id)

    def resolve(self):
        raise NotAvailable()

    def resolve_json(self):
        raise NotAvailable()

    def resolve_function_calls(self):
        raise NotAvailable()

    def __getstate__(self):
        """
        Support for pickling. Only store the call_id since that uniquely identifies the response.
        """
        return {"call_id": _call_to_concise_dict(self.call_id)}

    def __setstate__(self, state):
        """
        Support for unpickling. Restore the call_id, but value will need to be resolved later.
        """
        self.call_id = state["call_id"]
        self.value = None
        self._pr = None
