from typing import TYPE_CHECKING
from parallem.core.calls import _call_to_concise_dict
from parallem.core.exception import NotAvailable
from parallem.types import (
    CallIdentifier,
    LLMResponse,
    ParsedResponse,
)

if TYPE_CHECKING:
    from parallem.types import BaseRetriever


class PendingLLMResponse(LLMResponse):
    """
    Base class for a response from an LLM.
    """

    def __init__(
        self,
        call_id: CallIdentifier,
        backend: "BaseRetriever",
    ):
        super().__init__(value=None, call_id=call_id)
        self._backend = backend

    @property
    def final_answer(self) -> str:
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

    def __await__(self) -> str:
        "Async obtain response value"

        async def _await_response():
            if self.value is not None:
                return self.value

            backend = self._backend
            if backend is None or self.call_id is None:
                return self.final_answer

            pr = await backend.await_response(self.call_id)
            if pr is None:
                self.value = None
                self._pr = None
                return None

            self._pr = pr
            self.value = pr.text
            return self.value

        return _await_response().__await__()


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

    @property
    def final_answer(self):
        raise NotAvailable()

    @property
    def final_json(self):
        raise NotAvailable()

    @property
    def output_fcs(self):
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
