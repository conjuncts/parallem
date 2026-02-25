from collections import UserDict
from typing import TYPE_CHECKING

from parallellm.core.hydrate import hydrate_llm_response


if TYPE_CHECKING:
    from parallellm.core.backend import BaseBackend
    from parallellm.file_io.file_manager import FileManager


class NonMessageState:
    def __init__(self, fm: "FileManager", backend: "BaseBackend"):
        super().__init__()
        self._fm = fm
        self._backend = backend

    def __setitem__(self, key, value):
        # TODO: Right now this saves to disk every time, which is slow.
        # Move towards in-memory storage; upon context manager exit, update disk.
        self._fm.save_userdata(key, value)
        # self.data[key] = value

    def __getitem__(self, key):
        item = self._fm.load_userdata(key)

        # If the loaded data is an LLMResponse, inject the backend and hydrate
        item = hydrate_llm_response(item, self._backend)

        return item

    def get(self, key, default=None):
        try:
            return self[key]
        except FileNotFoundError:
            return default
