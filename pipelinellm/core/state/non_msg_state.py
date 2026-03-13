from collections import UserDict
from typing import TYPE_CHECKING, Optional
from PIL import Image

from pipelinellm.core.hydrate import hydrate_llm_response
from pipelinellm.core.memoize.operations import SetNonMsgItemOp
from pipelinellm.types import FunctionCallOutput, FunctionCallRequest, LLMResponse


if TYPE_CHECKING:
    from pipelinellm.core.backend import BaseBackend
    from pipelinellm.core.memoize.operations import OperationLog


class NonMessageState(UserDict):
    def __init__(self, backend: "BaseBackend"):
        super().__init__()
        self._backend = backend
        self._tracking_operations = False
        self._operation_log: Optional["OperationLog"] = None

    def _track_operation(self, operation):
        if self._tracking_operations and self._operation_log is not None:
            self._operation_log.record(operation)

    def _validate_value(self, value):
        if isinstance(
            value,
            (str, Image.Image, FunctionCallRequest, FunctionCallOutput, LLMResponse),
        ):
            return
        if isinstance(value, tuple):
            if (
                len(value) == 2
                and value[0] in ("user", "assistant", "system", "developer")
                and isinstance(value[1], str)
            ):
                return
        if isinstance(value, (int, float, bool, dict, list)):
            # OK (json method)
            return
        raise TypeError(
            f"NonMessageState values must be LLMDocument or LLMResponse, but got {type(value)}. "
            "Nested/container values are not supported."
        )

    def __setitem__(self, key, value):
        self._validate_value(value)
        self._track_operation(SetNonMsgItemOp(str(key), value))
        self.data[key] = value

    def __getitem__(self, key):
        item = self.data[key]

        # If the loaded data is an LLMResponse, inject the backend and hydrate
        item = hydrate_llm_response(item, self._backend)

        return item

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
