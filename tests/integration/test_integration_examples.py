"""
Integration tests combining multiple pllm features

These tests validate complex scenarios that combine:
- Tournaments
- Async/sync strategy switching
"""

import pytest
from uuid import uuid4
from parallem.core.gateway import resume_directory
from parallem.testing.simple_mock import mock_openai_client


@pytest.fixture
def test_agent_name(request):
    return f"{request.node.name}-{uuid4().hex}"


def test_tournament(shared_sync_orch, test_agent_name):
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

    mock_client = shared_sync_orch._mock_client
    mock_client.clear()
    mock_client.set_responses(responses)

    with shared_sync_orch.agent(test_agent_name) as agent:
        contestants_resp = agent.ask_llm("Get 4 contestants for the tournament")
        contestants = contestants_resp.final_answer.split("```")[1].split("\n")[1:5]

        semifinal_winners = []

        # Run semifinals
        for i in range(0, len(contestants), 2):
            resp = agent.ask_llm(f"Who wins: {contestants[i]} vs {contestants[i + 1]}?")
            semifinal_winners.append(resp.final_answer)

        final_resp = agent.ask_llm(
            f"Final match: {semifinal_winners[0]} vs {semifinal_winners[1]}?"
        )
        winner = final_resp.final_answer

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
    mock_client_async = mock_openai_client(responses=["Should not be called"], async_mode=True)
    with resume_directory(
        test_dir,
        provider="openai",
        strategy="async",  # Different strategy
        client=mock_client_async,
    ) as orch_async:
        # Add new data with async strategy
        with orch_async.agent() as agent:
            new_resp = agent.ask_llm("Test question")  # Should hit cache
            assert new_resp.final_answer == "Sync response"  # Cached from sync run

            with agent.memoize() as mem:
                mem.begin()  # load value
            assert (
                orch_async.userdata["sync_result"] == "Sync response"
            )  # Still there after loading in async

        # Verify no new API calls (cache hit)
        assert len(mock_client_async.calls) == 0


def test_complex_userdata_workflow(shared_sync_orch, test_agent_name):
    """Test complex userdata operations across multiple agents"""
    responses = [
        "Database schema v2.1",
        "Final implementation plan ready",
    ]

    mock_client = shared_sync_orch._mock_client
    mock_client.clear()
    mock_client.set_responses(responses)

    with shared_sync_orch.agent(f"{test_agent_name}-writer") as agent2:
        schema = agent2.ask_llm("Design database schema")
        shared_sync_orch.userdata["technical/database_schema"] = schema.final_answer

    with shared_sync_orch.agent(f"{test_agent_name}-reader") as agent3:  # noqa: F841
        db_schema = shared_sync_orch.userdata["technical/database_schema"]
        assert db_schema == "Database schema v2.1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
