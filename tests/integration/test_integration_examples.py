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
    with resume_directory(
        temp_integration_dir / "userdata_tournament",
        provider="openai",
        strategy="sync",
        client=mock_client,
    ) as orch:
        with orch.agent() as agent:
            contestants_resp = agent.ask_llm("Get 4 contestants for the tournament")
            contestants = contestants_resp.resolve().split("```")[1].split("\n")[1:5]

            semifinal_winners = []

            # Run semifinals
            for i in range(0, len(contestants), 2):
                resp = agent.ask_llm(
                    f"Who wins: {contestants[i]} vs {contestants[i + 1]}?"
                )
                semifinal_winners.append(resp.resolve())

            final_resp = agent.ask_llm(
                f"Final match: {semifinal_winners[0]} vs {semifinal_winners[1]}?"
            )
            winner = final_resp.resolve()

            assert winner == "Alice"
        assert len(mock_client.calls) == 4  # 1 contestants + 2 semifinals + 1 final


def test_strategy_switching_persistence(temp_integration_dir):
    """Test that data persists when switching between sync/async strategies"""
    test_dir = temp_integration_dir / "strategy_switch"

    # Run 1: Use sync strategy
    mock_client_sync = mock_openai_client(responses=["Sync response"])
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client_sync
    ) as orch_sync:
        with orch_sync.agent() as agent:
            resp = agent.ask_llm("Test question")
            assert resp.final_answer == "Sync response"

            with agent.memoize() as mem:
                mem.begin()
                orch_sync.userdata["sync_result"] = resp.final_answer

    assert len(mock_client_sync.calls) == 1

    # Run 2: Switch to async strategy, load same data
    mock_client_async = mock_openai_client(
        responses=["Should not be called"], concurrent=True
    )
    with resume_directory(
        test_dir,
        provider="openai",
        strategy="concurrent",  # Different strategy
        client=mock_client_async,
    ) as orch_async:
        # Add new data with async strategy
        with orch_async.agent() as agent:
            new_resp = agent.ask_llm("Test question")  # Should hit cache
            assert new_resp.resolve() == "Sync response"  # Cached from sync run

            with agent.memoize() as mem:
                mem.begin()  # load value
            assert (
                orch_async.userdata["sync_result"] == "Sync response"
            )  # Still there after loading in async

        # Verify no new API calls (cache hit)
        assert len(mock_client_async.calls) == 0


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
