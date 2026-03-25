def test_session_id_restoration(temp_integration_dir):
    """Test that sess_id is preserved when retrieving cached responses across sessions"""
    from parallem.core.gateway import resume_directory
    from parallem.testing.simple_mock import mock_openai_client

    test_dir = temp_integration_dir / "session_test"

    # Session 0: Store a response
    mock_client1 = mock_openai_client(responses=["Response from session 0"])
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client1
    ) as orch1:
        assert orch1.get_session_counter() == 0

        with orch1.agent() as agent:
            resp1 = agent.ask_llm("Test query")
            call_id_1 = resp1.call_id
            assert call_id_1["session_id"] == 0
            result1 = resp1.resolve()
            assert result1 == "Response from session 0"

    # Session 1: Retrieve cached response
    mock_client2 = mock_openai_client(responses=["Should not be called"])
    with resume_directory(
        test_dir, provider="openai", strategy="sync", client=mock_client2
    ) as orch2:
        assert orch2.get_session_counter() == 1

        with orch2.agent() as agent:
            resp2 = agent.ask_llm("Test query")  # Same query, should hit cache
            resp3 = agent.ask_llm("Test query")  # Same query again

            # The call_id should be restored to session_id=0 from cache
            assert resp2.call_id["session_id"] == 0
            assert resp3.call_id["session_id"] == 0

            result2 = resp2.resolve()
            assert result2 == "Response from session 0"

            # Check that seq_id is NOT sequential but in fact restored from cache
            assert resp2.call_id["seq_id"] == 0
            assert resp3.call_id["seq_id"] == 0  # !!

        # Verify no new API calls in session 1
        assert len(mock_client2.calls) == 0
