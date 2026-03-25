"""Simple fixtures for mocking OpenAI responses"""

from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class MockResponse:
    """Represents a mock response from an LLM"""

    output_text: str
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    response_id: Optional[str] = None

    def __post_init__(self):
        """Set default usage if not provided"""
        if self.usage is None:
            self.usage = {
                "prompt_tokens": 10,
                "completion_tokens": len(self.output_text.split()),
                "total_tokens": 10 + len(self.output_text.split()),
            }
        if self.response_id is None:
            self.response_id = "mock-response-id"
