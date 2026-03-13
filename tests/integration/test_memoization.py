"""
Integration tests for memoization feature

These tests validate that:
- Operations on MessageState are recorded and replayed correctly
- Salt parameter creates separate memoization caches
- LLM calls are avoided on replay
- Complex operation sequences work properly
"""

from pipelinellm.core.gateway import resume_directory
from pipelinellm.testing.simple_mock import mock_openai_client
import pytest


def test_basic_memoization(temp_integration_dir):
    """Test that operations are memoized and replayed on second run"""
    test_dir = temp_integration_dir / "basic_memoize"

    # First run: Execute computation and memoize
    mock_client1 = mock_openai_client(
        responses=["The number 42 is the answer to life, the universe, and everything."]
    )
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client1
    ) as orch1:
        with orch1.agent() as agent:
            conv = agent.get_msg_state()

            with agent.memoize() as mem:
                mem.begin()
                # Add some text
                conv.append("What is special about the number 42?")
                # Ask LLM
                conv.ask_llm()

            # Verify first run called the LLM
            assert len(mock_client1.calls) == 1
            assert "answer to life" in conv[-1].final_answer.lower()
            assert len(conv) == 2  # User message + LLM response

    # Second run: Should replay from memoization
    mock_client2 = mock_openai_client(responses=["Should not be called"])
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client2
    ) as orch2:
        with orch2.agent() as agent:
            conv = agent.get_msg_state()

            with agent.memoize() as mem:
                mem.begin()
                # Same operations
                conv.append("What is special about the number 42?")
                conv.ask_llm()

            # Verify second run did NOT call the LLM (replayed from memoize)
            assert len(mock_client2.calls) == 0
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


def test_memoization_complex_operations(temp_integration_dir):
    """Test that complex operation sequences are replayed correctly"""
    test_dir = temp_integration_dir / "complex_memoize"

    # First run: Complex sequence of operations
    mock_client1 = mock_openai_client(
        responses=[
            "First response",
            "Second response",
            "Third response",
        ]
    )
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client1
    ) as orch1:
        with orch1.agent() as agent:
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
            assert len(mock_client1.calls) == 3
            assert len(conv) == 6  # 3 questions + 3 responses
            assert conv[1].final_answer == "First response"
            assert conv[3].final_answer == "Second response"
            assert conv[5].final_answer == "Third response"

    # Second run: Should replay all operations
    mock_client2 = mock_openai_client(responses=["Should not be called"])
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client2
    ) as orch2:
        with orch2.agent() as agent:
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
            assert len(mock_client2.calls) == 0  # No new calls
            assert len(conv) == 6
            assert conv[1].final_answer == "First response"
            assert conv[3].final_answer == "Second response"
            assert conv[5].final_answer == "Third response"


def test_memoization_state_changes(temp_integration_dir):
    """Test that different initial states create different memoization caches"""
    test_dir = temp_integration_dir / "state_memoize"

    # First run with empty state
    mock_client1 = mock_openai_client(responses=["Response with empty state"])
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client1
    ) as orch1:
        with orch1.agent() as agent:
            conv = agent.get_msg_state()

            with agent.memoize() as mem:
                mem.begin()
                conv.append("Test")
                conv.ask_llm()

            assert len(mock_client1.calls) == 1
            assert conv[-1].final_answer == "Response with empty state"

    # Second run with pre-populated state - should NOT replay (different state hash)
    mock_client2 = mock_openai_client(responses=["Response with existing state"])
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client2
    ) as orch2:
        with orch2.agent() as agent:
            conv = agent.get_msg_state()

            # Pre-populate state
            conv.append("Some existing message")

            with agent.memoize() as mem:
                mem.begin()
                conv.append("Test")
                conv.ask_llm()

            # Should call LLM (different initial state)
            assert len(mock_client2.calls) == 1
            assert conv[-1].final_answer == "Response with existing state"


def test_memoization_with_no_operations(temp_integration_dir):
    """Test that memoization handles empty operation log gracefully"""
    test_dir = temp_integration_dir / "empty_memoize"

    # First run with no operations
    mock_client1 = mock_openai_client(responses=[])
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client1
    ) as orch1:
        with orch1.agent() as agent:
            conv = agent.get_msg_state()

            with agent.memoize() as mem:
                mem.begin()
                # No operations

            assert len(conv) == 0


def test_non_msg_state_memoized_replay(temp_integration_dir):
    test_dir = temp_integration_dir / "non_msg_memoize"

    mock_client1 = mock_openai_client(responses=["first answer"])
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client1
    ) as orch1:
        with orch1.agent() as agent:
            conv = agent.get_msg_state()

            with agent.memoize() as mem:
                mem.begin()
                orch1.userdata["saved_prompt"] = "What is memoization?"
                conv.append(orch1.userdata["saved_prompt"])
                conv.ask_llm()

            assert len(mock_client1.calls) == 1
            assert orch1.userdata["saved_prompt"] == "What is memoization?"

    mock_client2 = mock_openai_client(responses=["should not be called"])
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client2
    ) as orch2:
        with orch2.agent() as agent:
            conv = agent.get_msg_state()

            with agent.memoize() as mem:
                mem.begin()
                orch2.userdata["saved_prompt"] = "What is memoization?"
                conv.append(orch2.userdata["saved_prompt"])
                conv.ask_llm()

            assert len(mock_client2.calls) == 0
            assert orch2.userdata["saved_prompt"] == "What is memoization?"
            assert conv[-1].final_answer == "first answer"


def test_non_msg_state_rejects_nested_values(temp_integration_dir):
    test_dir = temp_integration_dir / "non_msg_type_guard"

    mock_client = mock_openai_client(responses=[])
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client
    ) as orch:
        with pytest.raises(TypeError):
            orch.userdata["nested"] = {"a": "b"}

        with pytest.raises(TypeError):
            orch.userdata["nested_list"] = ["x"]

    # Second run - should also have no operations
    mock_client2 = mock_openai_client(responses=[])
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client2
    ) as orch2:
        with orch2.agent() as agent:
            conv = agent.get_msg_state()

            with agent.memoize() as mem:
                mem.begin()
                # No operations

            assert len(conv) == 0
