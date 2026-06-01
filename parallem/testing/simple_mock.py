"""Simple testing utilities for mocking OpenAI calls"""

from dataclasses import asdict
from typing import List, Union, Optional, Dict
from unittest.mock import Mock
import asyncio
import re

from .fixtures import MockResponse


class MockOpenAIClient:
    """Mock Sync OpenAI client that returns predefined responses"""

    def __init__(self):
        self.calls = []
        self.response_list = []  # List of responses to return sequentially
        self.patterns = {}  # Regex patterns -> responses
        self.default = "Mock response"
        self.response_index = 0
        self.response_counter = 0  # Counter for generating unique response_ids

        mock = Mock()
        mock.create = self._create_response
        mock.parse = self._create_response
        self.responses = mock

    def set_responses(self, responses: List[Union[str, MockResponse]]):
        """Set responses to return sequentially"""
        self.response_list = [self._to_mock_response(r) for r in responses]
        self.patterns = {}
        self.response_index = 0

    def add_pattern(self, pattern: str, response: Union[str, MockResponse], *, literal=False):
        """Add regex pattern -> response mapping"""
        if literal:
            pattern = re.escape(pattern)
        self.patterns[pattern] = self._to_mock_response(response)

    def add_patterns(self, mapping: Dict[str, Union[str, MockResponse]], *, literal=False):
        """Add multiple patterns from a dict mapping"""
        for pattern, response in mapping.items():
            self.add_pattern(pattern, response, literal=literal)

    def set_default(self, response: Union[str, MockResponse]):
        """Set default response"""
        self.default = self._to_mock_response(response)

    def _to_mock_response(self, response):
        """Convert string to MockResponse if needed"""
        return MockResponse(output_text=response) if isinstance(response, str) else response

    def _get_input_text(self, input_messages):
        """Extract text from input messages"""
        if not input_messages:
            return ""

        parts = []
        for msg in input_messages:
            if isinstance(msg, str):
                parts.append(msg)
            elif isinstance(msg, dict) and "content" in msg:
                parts.append(msg["content"])
            elif hasattr(msg, "content"):
                parts.append(msg.content)
            else:
                parts.append(str(msg))

        return " ".join(parts)

    def _create_response(self, model=None, instructions=None, input=None, **kwargs):
        """Mock responses.create"""
        self.calls.append(
            {
                "model": model,
                "instructions": instructions,
                "input": input or [],
                "kwargs": kwargs,
            }
        )

        # Generate unique response_id
        unique_response_id = f"mock-response-{self.response_counter}"
        self.response_counter += 1

        # Sequential responses take priority
        if self.response_list and self.response_index < len(self.response_list):
            response = self.response_list[self.response_index]
            self.response_index += 1
            response_dict = asdict(response)
            response_dict["response_id"] = unique_response_id
            return response_dict

        # Check regex patterns
        text = f"{instructions or ''} {self._get_input_text(input or [])}"
        for pattern, response in self.patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                response_dict = asdict(response)
                response_dict["response_id"] = unique_response_id
                return response_dict

        # Default response
        response_dict = asdict(
            self.default
            if isinstance(self.default, MockResponse)
            else MockResponse(output_text=self.default, id=unique_response_id)
        )
        response_dict["response_id"] = unique_response_id
        return response_dict

    def clear(self):
        """Clear call history and reset responses"""
        self.calls = []
        self.response_list = []
        self.patterns = {}
        self.default = "Mock response"
        self.response_index = 0
        self.response_counter = 0


class MockConcurrentOpenAIClient(MockOpenAIClient):
    """Concurrent version of mock OpenAI client"""

    def __init__(self):
        super().__init__()
        self.responses.create = self._concurrent_create_response
        self.responses.parse = self._concurrent_create_response

    async def _concurrent_create_response(
        self, model=None, instructions=None, input=None, **kwargs
    ):
        """Mock concurrent responses.create"""
        result = self._create_response(model, instructions, input, **kwargs)
        await asyncio.sleep(0.01)  # Simulate concurrent delay
        return result


def mock_openai_client(
    responses: Optional[List[Union[str, MockResponse]]] = None,
    *,
    concurrent: bool = False,
) -> Union[MockOpenAIClient, MockConcurrentOpenAIClient]:
    """Create a mock OpenAI client for testing

    :param responses: List of responses to return sequentially
    :param concurrent: If True, return concurrent mock client
    :return: Mock client that can be passed to orchestrator constructor
    """
    if concurrent:
        mock_client = MockConcurrentOpenAIClient()
    else:
        mock_client = MockOpenAIClient()

    if responses:
        mock_client.set_responses(responses)

    return mock_client


# Assertion helpers
def assert_call_made(mock_client, content_substring: str):
    """Assert call containing substring was made"""
    for call in mock_client.calls:
        instructions = call.get("instructions", "") or ""
        input_text = mock_client._get_input_text(call.get("input", []))
        combined = f"{instructions} {input_text}".strip()

        if content_substring.lower() in combined.lower():
            return

    raise AssertionError(f"No call found containing '{content_substring}'")
