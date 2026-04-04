from parallem.core.gateway import resume_directory
from parallem.testing.simple_mock import mock_openai_client


def test_basic(temp_integration_dir):
    """Test that operations are memoized and replayed on second run"""
    test_dir = temp_integration_dir / "basic_memoize"

    # First run: Execute computation and memoize
    mock_client1 = mock_openai_client(
        responses=["The number 42 is the answer to life, the universe, and everything."]
    )
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client1
    ) as orch1:
        with orch1.agent("cross_context_agent") as agent:
            conv = agent.get_msg_state()

            conv.append("What is special about the number 42?")
            conv.ask_llm()
            assert len(mock_client1.calls) == 1
            assert "answer to life" in conv[-1].final_answer.lower()
            assert len(conv) == 2

        with orch1.agent("cross_context_agent") as agent:
            conv = agent.get_msg_state()
            assert len(conv) == 0  # Expect fresh conversation state
