from abc import ABC, abstractmethod
import threading
from typing import Dict, Tuple


class SeqIdStore(ABC):
    """Generate per-session sequence IDs for agents."""

    @abstractmethod
    def next_seq_id(self, session_id: int, agent_name: str) -> int:
        """
        Get the next sequence ID for an agent within a session.

        :param session_id: Unique session identifier.
        :param agent_name: Agent name.
        :return: Next sequence ID for this (session, agent) pair.
        """


class InMemorySeqIdStore(SeqIdStore):
    """In-memory sequence ID store scoped by session and agent name."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[Tuple[int, str], int] = {}

    def next_seq_id(self, session_id: int, agent_name: str) -> int:
        key = (session_id, agent_name)
        with self._lock:
            current = self._counters.get(key, 0)
            self._counters[key] = current + 1
            return current
