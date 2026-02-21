"""
Integration tests combining multiple pllm features

These tests validate complex scenarios that combine:
- Tournaments
- Async/sync strategy switching
- Userdata persistence across different workflows
"""

import pytest
from parallellm.core.gateway import resume_directory
from parallellm.testing.simple_mock import mock_openai_calls


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

    orch = resume_directory(
        temp_integration_dir / "checkpoint_tournament",
        provider="openai",
        strategy="sync",
    )

    mock_client = mock_openai_calls(orch, responses=responses)
    agent = orch.agent()

    # Phase 1: Get contestants (always runs)
    with agent:
        contestants_resp = agent.ask_llm("Get 4 contestants for the tournament")
        contestants = contestants_resp.resolve().split("```")[1].split("\n")[1:5]
        orch.save_userdata("contestants", contestants)

    # Phase 2: Semi-finals
    with agent:
        contestants = orch.load_userdata("contestants")
        semifinal_winners = []

        # Run semifinals
        for i in range(0, len(contestants), 2):
            resp = agent.ask_llm(f"Who wins: {contestants[i]} vs {contestants[i + 1]}?")
            semifinal_winners.append(resp.resolve())

        orch.save_userdata("semifinal_winners", semifinal_winners)

    # Phase 3: Finals
    with agent:
        finalists = orch.load_userdata("semifinal_winners")
        final_resp = agent.ask_llm(f"Final match: {finalists[0]} vs {finalists[1]}?")
        winner = final_resp.resolve()

        orch.save_userdata("tournament_winner", winner)

    # Verify results
    assert orch.load_userdata("tournament_winner") == "Alice"
    assert len(mock_client.calls) == 4  # 1 contestants + 2 semifinals + 1 final

    orch.persist()


def test_strategy_switching_persistence(temp_integration_dir):
    """Test that data persists when switching between sync/async strategies"""
    test_dir = temp_integration_dir / "strategy_switch"

    # Run 1: Use sync strategy
    pllm_sync = resume_directory(test_dir, provider="openai", strategy="sync")

    mock_client_sync = mock_openai_calls(pllm_sync, responses=["Sync response"])

    with pllm_sync.agent() as agent:
        resp = agent.ask_llm("Test question")
        result = resp.resolve()
        pllm_sync.save_userdata("sync_result", result)

    pllm_sync.persist()
    assert len(mock_client_sync.calls) == 1

    # Run 2: Switch to async strategy, load same data
    pllm_async = resume_directory(
        test_dir,
        provider="openai",
        strategy="async",  # Different strategy
    )

    mock_client_async = mock_openai_calls(
        pllm_async, responses=["Should not be called"]
    )

    # Should be able to load data created with sync strategy
    loaded_result = pllm_async.load_userdata("sync_result")
    assert loaded_result == "Sync response"

    # Add new data with async strategy
    with pllm_async.agent() as agent:
        new_resp = agent.ask_llm("Test question")  # Should hit cache
        assert new_resp.resolve() == "Sync response"  # Cached from sync run

        pllm_async.save_userdata("async_addition", "async_data")

    # Verify no new API calls (cache hit)
    assert len(mock_client_async.calls) == 0

    pllm_async.persist()


def test_complex_userdata_workflow(temp_integration_dir):
    """Test complex userdata operations across multiple agents"""
    responses = [
        "Project Alpha is the best",
        "Database schema v2.1",
        "Final implementation plan ready",
    ]

    orch = resume_directory(
        temp_integration_dir / "complex_userdata",
        provider="openai",
        strategy="sync",
    )

    mock_client = mock_openai_calls(orch, responses=responses)

    # Agent 1: Project selection
    agent1 = orch.agent()
    with agent1:
        project = agent1.ask_llm("Choose the best project")
        orch.save_userdata("selected_project", project.resolve())

    # Agent 2: Technical planning
    agent2 = orch.agent()
    with agent2:
        schema = agent2.ask_llm("Design database schema")
        orch.save_userdata("technical/database_schema", schema.resolve())

    # Agent 3: Final planning (uses data from both previous agents)
    agent3 = orch.agent()
    with agent3:
        project_name = orch.load_userdata("selected_project")
        db_schema = orch.load_userdata("technical/database_schema")

        plan = agent3.ask_llm(
            f"Create implementation plan for {project_name} with {db_schema}"
        )
        final_plan = plan.resolve()

        # Store hierarchical userdata
        orch.save_userdata("final/plan", final_plan)
        orch.save_userdata("final/project", project_name)
        orch.save_userdata("final/schema", db_schema)

    # Verify all userdata can be retrieved
    assert "Alpha" in orch.load_userdata("selected_project")
    assert "v2.1" in orch.load_userdata("technical/database_schema")
    assert "implementation plan" in orch.load_userdata("final/plan")

    assert len(mock_client.calls) == 3
    orch.persist()


def test_mixed_checkpoint_and_caching(temp_integration_dir):
    """Test interaction between checkpoints and caching"""
    test_dir = temp_integration_dir / "checkpoint_cache"

    # First run: Create checkpoints and cache
    pllm1 = resume_directory(test_dir, provider="openai", strategy="sync")

    mock_client1 = mock_openai_calls(
        pllm1, responses=["Initial data", "Checkpoint A result", "Checkpoint B result"]
    )

    agent1 = pllm1.agent()

    # Regular call (will be cached)
    with agent1:
        initial = agent1.ask_llm("Get initial data")
        pllm1.save_userdata("initial", initial.resolve())

    # Checkpoint A
    with agent1:
        result_a = agent1.ask_llm("Process A")
        pllm1.save_userdata("result_a", result_a.resolve())

    # Checkpoint B
    with agent1:
        result_b = agent1.ask_llm("Process B")
        pllm1.save_userdata("result_b", result_b.resolve())

    pllm1.persist()
    assert len(mock_client1.calls) == 3

    # Second run: Should use cache for non-checkpoint calls
    pllm2 = resume_directory(test_dir, provider="openai", strategy="sync")

    mock_client2 = mock_openai_calls(pllm2, responses=["Should not be called"])

    agent2 = pllm2.agent()

    # This should hit cache
    with agent2:
        initial2 = agent2.ask_llm("Get initial data")  # Same as before
        assert initial2.resolve() == "Initial data"  # From cache

    # Load checkpoint data
    assert pllm2.load_userdata("result_a") == "Checkpoint A result"
    assert pllm2.load_userdata("result_b") == "Checkpoint B result"

    # Verify no new API calls
    assert len(mock_client2.calls) == 0

    pllm2.persist()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
