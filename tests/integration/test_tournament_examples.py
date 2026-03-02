"""
Test tournament examples with both mocked and real functionality

These tests validate:
1. Tournament logic with mocked responses (fast)
2. File persistence and SQLite schema (with real directory operations)
3. Caching behavior on second runs
"""

import pytest
from parallellm.core.gateway import resume_directory
from parallellm.testing.simple_mock import mock_openai_calls


def test_nfl_tournament_mocked(temp_integration_dir):
    """Test NFL tournament with mocked responses"""
    # Set up mock responses
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

    orch = resume_directory(
        temp_integration_dir / "tour-nfl", provider="openai", strategy="sync"
    )

    mock_client = mock_openai_calls(orch, responses=responses)

    with orch.agent() as dash:
        # Get teams
        resp = dash.ask_llm(
            "Please name 8 NFL teams. Place your final answer in a code block, separated by newlines."
        )

        teams = resp.resolve().split("```")[1].split("\n")[1:9]
        assert len(teams) == 8
        assert "Patriots" in teams
        assert "Cowboys" in teams

        # Run games
        games = []
        for i in range(0, len(teams), 2):
            resp = dash.ask_llm(
                f"Given a game between the {teams[i]} and the {teams[i + 1]}, simply predict the winner and the score."
            )
            games.append(resp)

        # Get results
        game_descriptions = []
        for resp in games:
            game_descriptions.append(resp.resolve())

        assert len(game_descriptions) == 4
        assert "Patriots" in game_descriptions[0]
        assert "Packers" in game_descriptions[1]
        assert "Chiefs" in game_descriptions[2]
        assert "Ravens" in game_descriptions[3]

    # Verify mock was called correctly
    assert len(mock_client.calls) == 5  # 1 for teams + 4 for games

    # Persist and verify directory structure
    orch.persist()
    assert (temp_integration_dir / "tour-nfl").exists()


def test_enzyme_tournament_mocked(temp_integration_dir):
    """Test enzyme tournament with elimination rounds"""
    # Mock responses for a multi-round tournament
    responses = [
        # Initial enzymes
        """Here are 8 enzymes:
```
Amylase
Pepsin
Trypsin
Lipase
Catalase
Peroxidase
Lysozyme
Chymotrypsin
```""",
        # Round 1 (8 -> 4)
        "Amylase",  # Amylase vs Pepsin
        "Trypsin",  # Trypsin vs Lipase
        "Catalase",  # Catalase vs Peroxidase
        "Lysozyme",  # Lysozyme vs Chymotrypsin
        # Round 2 (4 -> 2)
        "Amylase",  # Amylase vs Trypsin
        "Catalase",  # Catalase vs Lysozyme
        # Final (2 -> 1)
        "Amylase",  # Amylase vs Catalase
    ]

    orch = resume_directory(
        temp_integration_dir / "tour-enzyme", provider="openai", strategy="sync"
    )

    mock_client = mock_openai_calls(orch, responses=responses)

    with orch.agent() as d:
        # Get initial enzymes
        resp = d.ask_llm(
            "Please name 8 enzymes. Place your final answer in a code block, separated by newlines."
        )

        teams = [x for x in resp.resolve().split("```")[1].split("\n")[1:] if x]
        assert len(teams) == 8

        # Tournament elimination
        while len(teams) > 1:
            responses_round = []
            for i in range(0, len(teams), 2):
                if i + 1 < len(teams):
                    resp = d.ask_llm(
                        "Given two enzymes, choose the one you like more. Only respond with the name of the enzyme.",
                        teams[i],
                        teams[i + 1],
                    )
                    responses_round.append(resp)
                else:
                    # Odd number - this team advances automatically
                    from parallellm.types import LLMResponse

                    responses_round.append(LLMResponse(teams[i]))

            # Resolve all responses for this round
            teams = [resp.resolve() for resp in responses_round]

        # Winner should be Amylase based on our mock responses
        assert len(teams) == 1
        assert teams[0] == "Amylase"

    # Verify correct number of API calls
    # 1 initial + 4 round1 + 2 round2 + 1 final = 8 calls
    assert len(mock_client.calls) == 8

    orch.persist()


def test_tournament_persistence_and_caching(temp_integration_dir):
    """Test that tournaments work with real persistence and caching"""
    tournament_dir = temp_integration_dir / "tour-persistence-test"

    # Mock the same responses for both runs
    responses = [
        """Teams:
```
Team A
Team B
Team C
Team D
```""",
        "Team A wins",
        "Team C wins",
    ]

    # First run - should make API calls
    pllm1 = resume_directory(tournament_dir, provider="openai", strategy="sync")

    mock_client1 = mock_openai_calls(pllm1, responses=responses)

    with pllm1.agent() as agent:
        teams_resp = agent.ask_llm("Get 4 teams")
        game1_resp = agent.ask_llm("Team A vs Team B")
        game2_resp = agent.ask_llm("Team C vs Team D")

        teams = teams_resp.resolve()
        game1 = game1_resp.resolve()
        game2 = game2_resp.resolve()

    pllm1.persist()

    # Verify API calls were made
    assert len(mock_client1.calls) == 3
    assert "Team A" in teams
    assert "Team A wins" in game1
    assert "Team C wins" in game2

    # Second run - should use cache (no API calls)
    pllm2 = resume_directory(tournament_dir, provider="openai", strategy="sync")

    mock_client2 = mock_openai_calls(pllm2, responses=["Should not be called"])

    with pllm2.agent() as agent:
        # Same queries should return cached results
        teams_resp2 = agent.ask_llm("Get 4 teams")
        game1_resp2 = agent.ask_llm("Team A vs Team B")
        game2_resp2 = agent.ask_llm("Team C vs Team D")

        teams2 = teams_resp2.resolve()
        game1_2 = game1_resp2.resolve()
        game2_2 = game2_resp2.resolve()

    pllm2.persist()

    # Verify no new API calls were made (cache hit)
    assert len(mock_client2.calls) == 0

    # Results should be identical
    assert teams == teams2
    assert game1 == game1_2
    assert game2 == game2_2


def test_tournament_with_concurrent_strategy(temp_integration_dir):
    """Test tournament works with concurrent strategy"""
    responses = [
        """Concurrent teams:
```
Alpha
Beta
Gamma
Delta
```""",
        "Alpha beats Beta",
        "Gamma beats Delta",
    ]

    orch = resume_directory(
        temp_integration_dir / "tour-concurrent-test",
        provider="openai",
        strategy="concurrent",  # Test concurrent strategy
    )

    mock_client = mock_openai_calls(orch, responses=responses)

    with orch.agent() as agent:
        teams_resp = agent.ask_llm("Get teams")

        # Submit multiple requests concurrently
        game_responses = []
        game_responses.append(agent.ask_llm("Alpha vs Beta"))
        game_responses.append(agent.ask_llm("Gamma vs Delta"))

        # Resolve all at once (tests async batching)
        teams = teams_resp.resolve()
        games = [resp.resolve() for resp in game_responses]

    assert "Alpha" in teams
    assert "Alpha beats Beta" in games[0]
    assert "Gamma beats Delta" in games[1]
    assert len(mock_client.calls) == 3

    orch.persist()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
