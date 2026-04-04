"""
Integration tests for memoization feature

These tests validate that:
- Operations on MessageState are recorded and replayed correctly
- Salt parameter creates separate memoization caches
- LLM calls are avoided on replay
- Complex operation sequences work properly
"""

from uuid import uuid4

import pytest

from parallem.core.gateway import resume_directory
from parallem.testing.simple_mock import mock_openai_client


@pytest.fixture
def test_agent_name(request):
    return f"{request.node.name}-{uuid4().hex}"


def test_basic_memoization(shared_sync_orch, test_agent_name):
    """Test that operations are memoized and replayed on second run"""
    mock_client = shared_sync_orch._mock_client

    # First run: Execute computation and memoize
    mock_client.clear()
    mock_client.set_responses(
        ["The number 42 is the answer to life, the universe, and everything."]
    )
    with shared_sync_orch.agent(test_agent_name) as agent:
        conv = agent.get_msg_state()

        with agent.memoize() as mem:
            mem.begin()
            # Add some text
            conv.append("What is special about the number 42?")
            # Ask LLM
            conv.ask_llm()

        # Verify first run called the LLM
        assert len(mock_client.calls) == 1
        assert "answer to life" in conv[-1].final_answer.lower()
        assert len(conv) == 2  # User message + LLM response

    # Second run: Should replay from memoization
    mock_client.clear()
    mock_client.set_responses(["Should not be called"])
    with shared_sync_orch.agent(test_agent_name) as agent:
        conv = agent.get_msg_state()

        with agent.memoize() as mem:
            mem.begin()
            # Same operations
            conv.append("What is special about the number 42?")
            conv.ask_llm()

        # Verify second run did NOT call the LLM (replayed from memoize)
        assert len(mock_client.calls) == 0
        assert "answer to life" in conv[-1].final_answer.lower()
        assert len(conv) == 2


def test_memoization_with_salt(temp_integration_dir):
    """Test that salt parameter creates separate memoization caches"""
    test_dir = temp_integration_dir / "salt_memoize"

    # First run with salt="v1"
    mock_client1 = mock_openai_client(responses=["Result for version 1"])
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client1
    ) as orch1:
        with orch1.agent() as agent:
            conv = agent.get_msg_state()

            with agent.memoize(salt="v1") as mem:
                mem.begin()
                conv.append("Test query")
                conv.ask_llm()

            assert len(mock_client1.calls) == 1
            assert conv[-1].final_answer == "Result for version 1"

    # Second run with salt="v2" - should NOT replay (different salt)
    mock_client2 = mock_openai_client(responses=["Result for version 2"])
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client2
    ) as orch2:
        with orch2.agent() as agent:
            conv = agent.get_msg_state()

            with agent.memoize(salt="v2") as mem:
                mem.begin()
                conv.append("Test query 2")  # Must be different (regular caching)
                conv.ask_llm()

            # Should call LLM (different salt)
            assert len(mock_client2.calls) == 1
            assert conv[-1].final_answer == "Result for version 2"

    # Third run with salt="v1" again - should replay from first run
    mock_client3 = mock_openai_client(responses=["Should not be called"])
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client3
    ) as orch3:
        with orch3.agent() as agent:
            conv = agent.get_msg_state()

            with agent.memoize(salt="v1") as mem:
                mem.begin()
                conv.append("Test query")
                conv.ask_llm()

            # Should NOT call LLM (replaying v1)
            assert len(mock_client3.calls) == 0
            assert conv[-1].final_answer == "Result for version 1"


def test_memoization_complex_operations(shared_sync_orch, test_agent_name):
    """Test that complex operation sequences are replayed correctly"""
    mock_client = shared_sync_orch._mock_client

    # First run: Complex sequence of operations
    mock_client.clear()
    mock_client.set_responses(
        [
            "First response",
            "Second response",
            "Third response",
        ]
    )
    with shared_sync_orch.agent(test_agent_name) as agent:
        conv = agent.get_msg_state()

        with agent.memoize() as mem:
            mem.begin()

            # Multiple appends
            conv.append("Question 1")
            conv.ask_llm()

            conv.append("Question 2")
            conv.ask_llm()

            # Extend with multiple items
            conv.extend(["Question 3"])
            conv.ask_llm()

        # Verify operations
        assert len(mock_client.calls) == 3
        assert len(conv) == 6  # 3 questions + 3 responses
        assert conv[1].final_answer == "First response"
        assert conv[3].final_answer == "Second response"
        assert conv[5].final_answer == "Third response"

    # Second run: Should replay all operations
    mock_client.clear()
    mock_client.set_responses(["Should not be called"])
    with shared_sync_orch.agent(test_agent_name) as agent:
        conv = agent.get_msg_state()

        with agent.memoize() as mem:
            mem.begin()

            # Same sequence
            conv.append("Question 1")
            conv.ask_llm()

            conv.append("Question 2")
            conv.ask_llm()

            conv.extend(["Question 3"])
            conv.ask_llm()

        # Verify replay
        assert len(mock_client.calls) == 0  # No new calls
        assert len(conv) == 6
        assert conv[1].final_answer == "First response"
        assert conv[3].final_answer == "Second response"
        assert conv[5].final_answer == "Third response"


def test_memoization_state_changes(shared_sync_orch, test_agent_name):
    """Test that different initial states create different memoization caches"""
    mock_client = shared_sync_orch._mock_client

    # First run with empty state
    mock_client.clear()
    mock_client.set_responses(["Response with empty state"])
    with shared_sync_orch.agent(test_agent_name) as agent:
        conv = agent.get_msg_state()

        with agent.memoize() as mem:
            mem.begin()
            conv.append("Test")
            conv.ask_llm()

        assert len(mock_client.calls) == 1
        assert conv[-1].final_answer == "Response with empty state"

    # Second run with pre-populated state - should NOT replay (different state hash)
    mock_client.clear()
    mock_client.set_responses(["Response with existing state"])
    with shared_sync_orch.agent(test_agent_name) as agent:
        conv = agent.get_msg_state()

        # Pre-populate state
        conv.append("Some existing message")

        with agent.memoize() as mem:
            mem.begin()
            conv.append("Test")
            conv.ask_llm()

        # Should call LLM (different initial state)
        assert len(mock_client.calls) == 1
        assert conv[-1].final_answer == "Response with existing state"


def test_memoization_with_no_operations(shared_sync_orch, test_agent_name):
    """Test that memoization handles empty operation log gracefully"""
    # First run with no operations
    mock_client = shared_sync_orch._mock_client
    mock_client.clear()
    mock_client.set_responses([])
    with shared_sync_orch.agent(test_agent_name) as agent:
        conv = agent.get_msg_state()

        with agent.memoize() as mem:
            mem.begin()
            # No operations

        assert len(conv) == 0


def test_non_msg_state_memoized_replay(shared_sync_orch, test_agent_name):
    mock_client = shared_sync_orch._mock_client

    mock_client.clear()
    mock_client.set_responses(["first answer"])
    with shared_sync_orch.agent(test_agent_name) as agent:
        conv = agent.get_msg_state()

        with agent.memoize() as mem:
            mem.begin()
            shared_sync_orch.userdata["saved_prompt"] = "What is memoization?"
            conv.append(shared_sync_orch.userdata["saved_prompt"])
            conv.ask_llm()

        assert len(mock_client.calls) == 1
        assert shared_sync_orch.userdata["saved_prompt"] == "What is memoization?"

    mock_client.clear()
    mock_client.set_responses(["should not be called"])
    with shared_sync_orch.agent(test_agent_name) as agent:
        conv = agent.get_msg_state()

        with agent.memoize() as mem:
            mem.begin()
            shared_sync_orch.userdata["saved_prompt"] = "What is memoization?"
            conv.append(shared_sync_orch.userdata["saved_prompt"])
            conv.ask_llm()

        assert len(mock_client.calls) == 0
        assert shared_sync_orch.userdata["saved_prompt"] == "What is memoization?"
        assert conv[-1].final_answer == "first answer"


def test_non_msg_state_accepts_nested_json_values(shared_sync_orch, test_agent_name):

    nested_value = {"a": "b", "nested": {"x": 1}, "items": ["x", {"y": 2}]}
    nested_list = ["x", {"k": "v"}, [1, 2, 3]]

    mock_client = shared_sync_orch._mock_client

    mock_client.clear()
    mock_client.set_responses(["nested accepted"])
    with shared_sync_orch.agent(test_agent_name) as agent:
        conv = agent.get_msg_state()

        with agent.memoize() as mem:
            mem.begin()
            shared_sync_orch.userdata["nested"] = nested_value
            shared_sync_orch.userdata["nested_list"] = nested_list
            conv.append(shared_sync_orch.userdata["nested"]["a"])
            conv.ask_llm()

        assert shared_sync_orch.userdata["nested"] == nested_value
        assert shared_sync_orch.userdata["nested_list"] == nested_list
        assert len(mock_client.calls) == 1
        assert conv[-1].final_answer == "nested accepted"

    mock_client.clear()
    mock_client.set_responses(["should not be called"])
    with shared_sync_orch.agent(test_agent_name) as agent:
        conv = agent.get_msg_state()

        with agent.memoize() as mem:
            mem.begin()
            shared_sync_orch.userdata["nested"] = nested_value
            shared_sync_orch.userdata["nested_list"] = nested_list
            conv.append(shared_sync_orch.userdata["nested"]["a"])
            conv.ask_llm()

        assert shared_sync_orch.userdata["nested"] == nested_value
        assert shared_sync_orch.userdata["nested_list"] == nested_list
        assert len(mock_client.calls) == 0
        assert conv[-1].final_answer == "nested accepted"
