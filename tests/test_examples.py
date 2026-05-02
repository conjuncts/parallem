from parallem.testing.simple_mock import (
    assert_call_made,
)


def test_simple_mock_responses(shared_sync_orch):
    """Test with a simple list of mock responses"""
    responses = ["First response", "Second response", "Third response"]

    mock_client = shared_sync_orch._mock_client
    mock_client.clear()
    mock_client.set_responses(responses)

    with shared_sync_orch.agent("test_simple_mock_responses") as a:
        resp1 = a.ask_llm("First question")
        resp2 = a.ask_llm("Second question")
        resp3 = a.ask_llm("Third question")

    # Check responses
    assert resp1.final_answer == "First response"
    assert resp2.final_answer == "Second response"
    assert resp3.final_answer == "Third response"

    # Check calls were recorded
    assert len(mock_client.calls) == 3
    assert_call_made(mock_client, "First question")
    assert_call_made(mock_client, "Second question")
    assert_call_made(mock_client, "Third question")


def test_pattern_based_responses(shared_sync_orch):
    """Test with pattern-based response mapping"""
    mock_client = shared_sync_orch._mock_client

    # Add patterns using the convenient dict method
    mock_client.clear()
    mock_client.add_patterns(
        {
            "calculate": "The answer is 42",
            "weather": "It's sunny today",
            "joke": "Why did the chicken cross the road? To get to the other side!",
        }
    )
    mock_client.set_default("Mock response for unknown question")

    with shared_sync_orch.agent("test_pattern_based_responses") as a:
        # These should match patterns
        calc_resp = a.ask_llm("Please calculate 2 + 2")
        weather_resp = a.ask_llm("What's the weather like?")
        joke_resp = a.ask_llm("Tell me a joke")

        # This won't match any pattern, so gets default response
        other_resp = a.ask_llm("Random question")

    assert "42" in calc_resp.final_answer
    assert "sunny" in weather_resp.final_answer
    assert "chicken" in joke_resp.final_answer
    assert "Mock response" in other_resp.final_answer  # Default response

    assert len(mock_client.calls) == 4


def test_exact_instruction_matching(shared_sync_orch):
    """Test with exact instruction matching"""
    mock_client = shared_sync_orch._mock_client

    # Add exact matches using the dict method with literal=True
    mock_client.clear()
    mock_client.add_patterns(
        {
            "What is the capital of France?": "The capital of France is Paris.",
            "What is 2 + 2?": "2 + 2 equals 4.",
        },
        literal=True,
    )
    mock_client.set_default("I don't know that.")

    with shared_sync_orch.agent("test_exact_instruction_matching") as a:
        resp1 = a.ask_llm("What is the capital of France?")
        resp2 = a.ask_llm("What is 2 + 2?")
        resp3 = a.ask_llm("What is the meaning of life?")

    assert "Paris" in resp1.final_answer
    assert "equals 4" in resp2.final_answer
    assert "don't know" in resp3.final_answer


def test_mixed_pattern_methods(shared_sync_orch):
    """Test mixing individual add_pattern and batch add_patterns"""
    mock_client = shared_sync_orch._mock_client

    # Add batch patterns first
    mock_client.clear()
    mock_client.add_patterns(
        {"math|calculate": "Math result: 42", "weather": "It's sunny"}
    )

    # Add individual pattern
    mock_client.add_pattern("greeting|hello", "Hello there!")

    # Set default
    mock_client.set_default("Default response")

    with shared_sync_orch.agent("test_mixed_pattern_methods") as a:
        math_resp = a.ask_llm("Calculate 2+2")
        weather_resp = a.ask_llm("What's the weather?")
        greeting_resp = a.ask_llm("Hello world")
        other_resp = a.ask_llm("Random question")

    assert "Math result: 42" in math_resp.final_answer
    assert "sunny" in weather_resp.final_answer
    assert "Hello there!" in greeting_resp.final_answer
    assert "Default response" in other_resp.final_answer

    assert len(mock_client.calls) == 4


def test_concurrent_provider(shared_concurrent_orch):
    """Test with concurrent provider"""
    responses = ["Async response 1", "Async response 2"]

    mock_client = shared_concurrent_orch._mock_client
    mock_client.clear()
    mock_client.set_responses(responses)

    with shared_concurrent_orch.agent("test_concurrent_provider") as a:
        resp1 = a.ask_llm("First async question")
        resp2 = a.ask_llm("Second async question")

    # Responses should resolve correctly
    assert resp1.final_answer == "Async response 1"
    assert resp2.final_answer == "Async response 2"

    assert len(mock_client.calls) == 2


def example_nfl_tournament_test(shared_sync_orch):
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

    mock_client = shared_sync_orch._mock_client
    mock_client.clear()
    mock_client.set_responses(responses)
    with shared_sync_orch.agent("example_nfl_tournament_test") as a:
        # Get teams
        resp = a.ask_llm(
            "Please name 8 NFL teams. Place your final answer in a code block, separated by newlines."
        )

        teams = resp.final_answer.split("```")[1].split("\n")[1:9]
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
            game_descriptions.append(resp.final_answer)

        print(f"Game results: {game_descriptions}")

    # Verify the mock worked as expected
    assert len(mock_client.calls) == 5  # 1 for teams + 4 for games
    assert len(game_descriptions) == 4
    assert "Patriots" in game_descriptions[0]
    assert "Packers" in game_descriptions[1]


if __name__ == "__main__":
    # Run the example
    import parallem as pllm

    print("Running NFL tournament test example...")
    with pllm.resume_directory(
        "temp_example_nfl_tournament_test", provider="openai", strategy="sync"
    ) as temp_orch:
        example_nfl_tournament_test(temp_orch)
    print("✓ NFL tournament test completed successfully!")

    # Run pytest tests
    print("\nTo run all tests, use: pytest tests/examples.py -v")
