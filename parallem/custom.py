from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

from parallem.types import AskItem

if TYPE_CHECKING:
    from hashlib import _Hash

class CustomAskItem(AskItem, ABC):

    type: str # Needs to be set

    @abstractmethod
    def compute_hash(self, hasher: "_Hash"):
        """
        Compute a hash for this custom AskItem.
        """

    @abstractmethod
    def to_chat_completion(self) -> Dict:
        """
        Convert this custom AskItem to OpenAI's ChatCompletion format.
        """

    @abstractmethod
    def to_provider(self, provider_type: str) -> Any:
        """
        Since provider cannot be expected to know how to handle custom documents,
        """
