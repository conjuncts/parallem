"""
Example usage of the simple ParalleLLM testing utilities

This module demonstrates how to use the simple mock utilities
to write effective tests for code that uses ParalleLLM with pytest.

For more comprehensive examples testing real ParalleLLM functionality:
- test_tournament_examples.py: Tests tournament logic, persistence, and caching
"""

import tempfile
import pytest
from parallellm.core.gateway import resume_directory
from parallellm.testing.simple_mock import (
    mock_openai_calls,
    assert_call_made,
)


@pytest.fixture
def temp_orch(tmp_path):
    """Pytest fixture that provides a temporary ParalleLLM instance"""
    orch_dir = tmp_path / "sync"
    orch = resume_directory(
        str(orch_dir), provider="openai", strategy="sync", client=None
    )
    yield orch


@pytest.fixture
def concurrent_temp_orch(tmp_path):
    """Pytest fixture that provides a temporary concurrent ParalleLLM instance"""
    orch_dir = tmp_path / "concurrent"
    orch = resume_directory(
        str(orch_dir), provider="openai", strategy="concurrent", client=None
    )
    yield orch


def test_simple_mock_responses(temp_orch):
    """Test with a simple list of mock responses"""
    responses = ["First response", "Second response", "Third response"]

    mock_client = mock_openai_calls(temp_orch, responses=responses)

    with temp_orch.agent() as a:
        resp1 = a.ask_llm("First question")
        resp2 = a.ask_llm("Second question")
        resp3 = a.ask_llm("Third question")

    # Check responses
    assert resp1.resolve() == "First response"
    assert resp2.resolve() == "Second response"
    assert resp3.resolve() == "Third response"

    # Check calls were recorded
    assert len(mock_client.calls) == 3
    assert_call_made(mock_client, "First question")
    assert_call_made(mock_client, "Second question")
    assert_call_made(mock_client, "Third question")


def test_pattern_based_responses(temp_orch):
    """Test with pattern-based response mapping"""
    mock_client = mock_openai_calls(temp_orch)

    # Add patterns using the convenient dict method
    mock_client.add_patterns(
        {
            "calculate": "The answer is 42",
            "weather": "It's sunny today",
            "joke": "Why did the chicken cross the road? To get to the other side!",
        }
    )
    mock_client.set_default("Mock response for unknown question")

    with temp_orch.agent() as a:
        # These should match patterns
        calc_resp = a.ask_llm("Please calculate 2 + 2")
        weather_resp = a.ask_llm("What's the weather like?")
        joke_resp = a.ask_llm("Tell me a joke")

        # This won't match any pattern, so gets default response
        other_resp = a.ask_llm("Random question")

    assert "42" in calc_resp.resolve()
    assert "sunny" in weather_resp.resolve()
    assert "chicken" in joke_resp.resolve()
    assert "Mock response" in other_resp.resolve()  # Default response

    assert len(mock_client.calls) == 4


def test_exact_instruction_matching(temp_orch):
    """Test with exact instruction matching"""
    mock_client = mock_openai_calls(temp_orch)

    # Add exact matches using the dict method with literal=True
    mock_client.add_patterns(
        {
            "What is the capital of France?": "The capital of France is Paris.",
            "What is 2 + 2?": "2 + 2 equals 4.",
        },
        literal=True,
    )
    mock_client.set_default("I don't know that.")

    with temp_orch.agent() as a:
        resp1 = a.ask_llm("What is the capital of France?")
        resp2 = a.ask_llm("What is 2 + 2?")
        resp3 = a.ask_llm("What is the meaning of life?")

    assert "Paris" in resp1.resolve()
    assert "equals 4" in resp2.resolve()
    assert "don't know" in resp3.resolve()


def test_mixed_pattern_methods(temp_orch):
    """Test mixing individual add_pattern and batch add_patterns"""
    mock_client = mock_openai_calls(temp_orch)

    # Add batch patterns first
    mock_client.add_patterns(
        {"math|calculate": "Math result: 42", "weather": "It's sunny"}
    )

    # Add individual pattern
    mock_client.add_pattern("greeting|hello", "Hello there!")

    # Set default
    mock_client.set_default("Default response")

    with temp_orch.agent() as a:
        math_resp = a.ask_llm("Calculate 2+2")
        weather_resp = a.ask_llm("What's the weather?")
        greeting_resp = a.ask_llm("Hello world")
        other_resp = a.ask_llm("Random question")

    assert "Math result: 42" in math_resp.resolve()
    assert "sunny" in weather_resp.resolve()
    assert "Hello there!" in greeting_resp.resolve()
    assert "Default response" in other_resp.resolve()

    assert len(mock_client.calls) == 4


def test_concurrent_provider(concurrent_temp_orch):
    """Test with concurrent provider"""
    responses = ["Async response 1", "Async response 2"]

    mock_client = mock_openai_calls(concurrent_temp_orch, responses=responses)

    with concurrent_temp_orch.agent() as a:
        resp1 = a.ask_llm("First async question")
        resp2 = a.ask_llm("Second async question")

    # Responses should resolve correctly
    assert resp1.resolve() == "Async response 1"
    assert resp2.resolve() == "Async response 2"

    assert len(mock_client.calls) == 2


def example_nfl_tournament_test():
    """Example of testing the NFL tournament from the examples"""

    # Set up responses for the tournament
    responses = [
        # Teams response
        """Here are 8 NFL teams:
```
Patriots
Cowboys  
Packers
49ers
Chiefs
Bills
Ravens
Steelers
```""",
        # Game predictions
        "Patriots beat Cowboys 21-14",
        "Packers defeat 49ers 28-21",
        "Chiefs beat Bills 31-17",
        "Ravens defeat Steelers 24-10",
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        orch = resume_directory(
            temp_dir, provider="openai", strategy="sync", client=None
        )
        mock_client = mock_openai_calls(orch, responses=responses)

        with orch.agent() as a:
            # Get teams
            resp = a.ask_llm(
                "Please name 8 NFL teams. Place your final answer in a code block, separated by newlines."
            )

            teams = resp.resolve().split("```")[1].split("\n")[1:9]
            print(f"Teams: {teams}")

            # Run games
            games = []
            for i in range(0, len(teams), 2):
                resp = a.ask_llm(
                    f"Given a game between the {teams[i]} and the {teams[i + 1]}, simply predict the winner and the score."
                )
                games.append(resp)

            # Get results
            game_descriptions = []
            for resp in games:
                game_descriptions.append(resp.resolve())

            print(f"Game results: {game_descriptions}")

        # Verify the mock worked as expected
        assert len(mock_client.calls) == 5  # 1 for teams + 4 for games
        assert len(game_descriptions) == 4
        assert "Patriots" in game_descriptions[0]
        assert "Packers" in game_descriptions[1]


if __name__ == "__main__":
    # Run the example
    print("Running NFL tournament test example...")
    example_nfl_tournament_test()
    print("✓ NFL tournament test completed successfully!")

    # Run pytest tests
    print("\nTo run all tests, use: pytest tests/examples.py -v")
