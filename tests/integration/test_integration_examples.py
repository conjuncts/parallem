"""
Integration tests combining multiple pllm features

These tests validate complex scenarios that combine:
- Tournaments
- Async/sync strategy switching
"""

import pytest
from pipelinellm.core.gateway import resume_directory
from pipelinellm.testing.simple_mock import mock_openai_client


def test_tournament(temp_integration_dir):
    """Test tournament"""
    responses = [
        # Phase 1: Get contestants
        """Contest participants:
```
Alice
Bob
Charlie
Diana
```""",
        # Phase 2: Elimination rounds
        "Alice",  # Alice vs Bob
        "Diana",  # Charlie vs Diana
        # Phase 3: Final round
        "Alice",  # Alice vs Diana
    ]

    mock_client = mock_openai_client(responses=responses)
    orch = resume_directory(
        temp_integration_dir / "checkpoint_tournament",
        provider="openai",
        strategy="sync",
        client=mock_client,
    )
    agent = orch.agent()

    # Phase 1: Get contestants (always runs)
    with agent:
        contestants_resp = agent.ask_llm("Get 4 contestants for the tournament")
        contestants = contestants_resp.resolve().split("```")[1].split("\n")[1:5]
        orch.userdata["contestants"] = contestants

    # Phase 2: Semi-finals
    with agent:
        contestants = orch.userdata["contestants"]
        assert len(contestants) == 4
        semifinal_winners = []

        # Run semifinals
        for i in range(0, len(contestants), 2):
            resp = agent.ask_llm(f"Who wins: {contestants[i]} vs {contestants[i + 1]}?")
            semifinal_winners.append(resp.resolve())

        orch.userdata["semifinal_winners"] = semifinal_winners

    # Phase 3: Finals
    with agent:
        finalists = orch.userdata["semifinal_winners"]
        assert len(finalists) == 2
        final_resp = agent.ask_llm(f"Final match: {finalists[0]} vs {finalists[1]}?")
        winner = final_resp.resolve()

        orch.userdata["tournament_winner"] = winner

    # Verify results
    assert orch.userdata["tournament_winner"] == "Alice"
    assert len(mock_client.calls) == 4  # 1 contestants + 2 semifinals + 1 final

    orch.persist()


def test_strategy_switching_persistence(temp_integration_dir):
    """Test that data persists when switching between sync/async strategies"""
    test_dir = temp_integration_dir / "strategy_switch"

    # Run 1: Use sync strategy
    mock_client_sync = mock_openai_client(responses=["Sync response"])
    orch_sync = resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client_sync
    )

    with orch_sync.agent() as agent:
        resp = agent.ask_llm("Test question")
        result = resp.resolve()
        orch_sync.userdata["sync_result"] = result

    orch_sync.persist()
    assert len(mock_client_sync.calls) == 1

    # Run 2: Switch to async strategy, load same data
    mock_client_async = mock_openai_client(
        responses=["Should not be called"], concurrent=True
    )
    orch_async = resume_directory(
        test_dir,
        provider="openai",
        strategy="concurrent",  # Different strategy
        client=mock_client_async,
    )

    # Should be able to load data created with sync strategy
    loaded_result = orch_async.userdata["sync_result"]
    assert loaded_result == "Sync response"

    # Add new data with async strategy
    with orch_async.agent() as agent:
        new_resp = agent.ask_llm("Test question")  # Should hit cache
        assert new_resp.resolve() == "Sync response"  # Cached from sync run

        orch_async.userdata["concurrent_addition"] = "concurrent_data"

    # Verify no new API calls (cache hit)
    assert len(mock_client_async.calls) == 0

    orch_async.persist()


def test_complex_userdata_workflow(temp_integration_dir):
    """Test complex userdata operations across multiple agents"""
    responses = [
        "Database schema v2.1",
        "Final implementation plan ready",
    ]

    mock_client = mock_openai_client(responses=responses)
    with resume_directory(
        temp_integration_dir / "complex_userdata",
        provider="openai",
        strategy="sync",
        client=mock_client,
    ) as orch:
        with orch.agent("2") as agent2:
            schema = agent2.ask_llm("Design database schema")
            orch.userdata["technical/database_schema"] = schema.resolve()

        with orch.agent("3") as agent3:  # noqa: F841
            db_schema = orch.userdata["technical/database_schema"]
            assert db_schema == "Database schema v2.1"


def test_mixed_checkpoint_and_caching(temp_integration_dir):
    """Test interaction between checkpoints and caching"""
    test_dir = temp_integration_dir / "checkpoint_cache"

    # First run: Create checkpoints and cache
    mock_client1 = mock_openai_client(
        responses=["Initial data", "Checkpoint A result", "Checkpoint B result"]
    )
    orch = resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client1
    )

    agent1 = orch.agent()

    # Regular call (will be cached)
    with agent1:
        initial = agent1.ask_llm("Get initial data")
        orch.userdata["initial"] = initial.resolve()

    # Checkpoint A
    with agent1:
        result_a = agent1.ask_llm("Process A")
        orch.userdata["result_a"] = result_a.resolve()

    # Checkpoint B
    with agent1:
        result_b = agent1.ask_llm("Process B")
        orch.userdata["result_b"] = result_b.resolve()

    orch.persist()
    assert len(mock_client1.calls) == 3

    # Second run: Should use cache for non-checkpoint calls
    mock_client2 = mock_openai_client(responses=["Should not be called"])
    orch2 = resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client2
    )

    agent2 = orch2.agent()

    # This should hit cache
    with agent2:
        initial2 = agent2.ask_llm("Get initial data")  # Same as before
        assert initial2.resolve() == "Initial data"  # From cache

    # Load checkpoint data
    assert orch2.userdata["result_a"] == "Checkpoint A result"
    assert orch2.userdata["result_b"] == "Checkpoint B result"

    # Verify no new API calls
    assert len(mock_client2.calls) == 0

    orch2.persist()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
