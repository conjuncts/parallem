"""Simple fixtures for mocking OpenAI responses"""

from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class MockResponse:
    """Represents a mock response from an LLM, in OpenAI format."""

    output_text: str
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    id: Optional[str] = None
    "Response ID"

    def __post_init__(self):
        """Set default usage if not provided"""
        if self.usage is None:
            self.usage = {
                "prompt_tokens": 10,
                "completion_tokens": len(self.output_text.split()),
                "total_tokens": 10 + len(self.output_text.split()),
            }
        if self.id is None:
            self.id = "mock-response-id"
