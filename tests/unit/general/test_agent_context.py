"""
Unit tests for AgentContext class

Tests the core agent functionality including:
- Context manager behavior (__enter__, __exit__)
- Counter systems (anonymous)
- ask_llm method and cache integration
- Exception handling
"""

import pytest
from unittest.mock import patch
from parallem.core.agent.agent import AgentContext
from parallem.core.exception import NotAvailable
from parallem.core.response import ReadyLLMResponse, PendingLLMResponse
from parallem.types import HumanResponse, LLMIdentity, ParsedResponse


class TestAgentContextBasics:
    """Test basic AgentContext functionality"""

    def test_context_manager_enter_exit(self, mock_orchestrator):
        """Test context manager protocol"""
        agent = AgentContext("test_agent", mock_orchestrator)

        # Should work without exception
        with agent:
            pass

    def test_exit_with_parallem_exceptions(self, mock_orchestrator):
        """Test __exit__ handles parallem exceptions correctly"""
        agent = AgentContext("test_agent", mock_orchestrator)

        suppressed_exceptions = [NotAvailable]

        for exc_type in suppressed_exceptions:
            with agent:
                raise exc_type("test exception")
        # Exceptions should have been suppressed

    def test_exit_with_other_exceptions(self, mock_orchestrator):
        """Test __exit__ doesn't suppress other exceptions"""
        agent = AgentContext("test_agent", mock_orchestrator)

        # Other exceptions should not be suppressed
        with pytest.raises(ValueError):
            with agent:
                raise ValueError("non-parallem exception")

        with pytest.raises(AssertionError):
            with agent:
                raise AssertionError("Should not suppress asserts!")


def test_counter_independence(mock_orchestrator):
    """Test that sequence ID counter works correctly"""
    agent = AgentContext("test_agent", mock_orchestrator)

    with agent:
        agent.ask_llm("anonymous 1")
        agent.ask_llm("anonymous 2")
        seq_ids = [
            call.kwargs["call_id"]["seq_id"]
            for call in mock_orchestrator._backend.submit_query.call_args_list
        ]
        assert seq_ids == [0, 1]

    mock_orchestrator._backend.submit_query.reset_mock()
    agent2 = AgentContext("test_agent", mock_orchestrator)
    with agent2:
        agent2.ask_llm("anonymous 1")
        call_id = mock_orchestrator._backend.submit_query.call_args.kwargs["call_id"]
        assert call_id["seq_id"] == 2

    agent3 = AgentContext("other_agent", mock_orchestrator)
    with agent3:
        agent3.ask_llm("anonymous 1")
        call_id = mock_orchestrator._backend.submit_query.call_args.kwargs["call_id"]
        assert call_id["seq_id"] == 0


class TestAgentContextCoercion:
    """Test coercion methods in AgentContext"""

    def test_coerce_tools_single_dict(self, mock_orchestrator):
        agent = AgentContext("test", mock_orchestrator)
        tool = {"name": "test_tool", "parameters": {}}
        coerced = agent._coerce_tools(tool)
        assert coerced == [tool]

    def test_coerce_tools_list_mixed(self, mock_orchestrator):
        agent = AgentContext("test", mock_orchestrator)

        def my_tool(x: int):
            """My tool description"""
            return x

        dict_tool = {"name": "dict_tool", "parameters": {}}
        tools = [dict_tool, my_tool]

        coerced = agent._coerce_tools(tools)

        assert len(coerced) == 2
        assert coerced[0] == dict_tool
        # to_tool_schema returns a LIST of schemas
        tool_schema = coerced[1]
        assert tool_schema["type"] == "function"
        assert tool_schema["name"] == "my_tool"
        assert "parameters" in tool_schema

    def test_coerce_tools_invalid_type(self, mock_orchestrator):
        agent = AgentContext("test", mock_orchestrator)
        with pytest.raises(ValueError, match="is not a dict, ServerTool, or callable"):
            agent._coerce_tools(123)


class TestAskLLMMethod:
    """Test the ask_llm method functionality"""

    def test_ask_llm_default_precedence(self, mock_orchestrator):
        """ask_llm precedence should be global default < ask_params < explicit kwargs."""
        global_default_agent = AgentContext("test_agent", mock_orchestrator)

        with global_default_agent:
            global_default_agent.ask_llm("global default prompt")
            global_params = mock_orchestrator._backend.submit_query.call_args.args[1]
            assert global_params["llm"].identity == "gpt-5-nano"

        mock_orchestrator._backend.submit_query.reset_mock()

        ask_params_agent = AgentContext(
            "test_agent",
            mock_orchestrator,
            ask_params={
                "llm": LLMIdentity("gpt-4o-mini", provider_type="openai"),
            },
        )
        with ask_params_agent:
            ask_params_agent.ask_llm("ask_params default prompt")
            ask_params_defaults = (
                mock_orchestrator._backend.submit_query.call_args.args[1]
            )
            assert ask_params_defaults["llm"].identity == "gpt-4o-mini"

        mock_orchestrator._backend.submit_query.reset_mock()

        with ask_params_agent:
            ask_params_agent.ask_llm(
                "explicit override prompt",
                llm=LLMIdentity("gpt-5-mini", provider_type="openai"),
            )
            explicit_override = mock_orchestrator._backend.submit_query.call_args.args[
                1
            ]
            assert explicit_override["llm"].identity == "gpt-5-mini"

    def test_msg_state_ask_llm_default_precedence(self, mock_orchestrator):
        """MessageState.ask_llm should inherit ask_params defaults and allow explicit overrides."""
        agent = AgentContext(
            "test_agent",
            mock_orchestrator,
            ask_params={
                "llm": LLMIdentity("gpt-4o-mini", provider_type="openai"),
            },
        )

        with agent:
            msg_state = agent.get_msg_state()
            msg_state.ask_llm("msg_state default prompt")
            ask_params_defaults = (
                mock_orchestrator._backend.submit_query.call_args.args[1]
            )
            assert ask_params_defaults["llm"].identity == "gpt-4o-mini"

        mock_orchestrator._backend.submit_query.reset_mock()

        with agent:
            msg_state = agent.get_msg_state()
            msg_state.ask_llm(
                "msg_state explicit override prompt",
                llm=LLMIdentity("gpt-5-mini", provider_type="openai"),
            )
            explicit_override = mock_orchestrator._backend.submit_query.call_args.args[
                1
            ]
            assert explicit_override["llm"].identity == "gpt-5-mini"

    def test_ask_llm_callable_tool_coercion(self, mock_orchestrator):
        """Test that passing a callable to ask_llm tools is correctly coerced to a schema."""
        agent = AgentContext("test_agent", mock_orchestrator)

        def get_weather(city: str):
            """Get weather for a city"""
            return f"Sunny in {city}"

        with agent:
            agent.ask_llm("What is the weather in NYC?", tools=[get_weather])

            # Check what was passed to submit_query
            submit_args = mock_orchestrator._backend.submit_query.call_args.args
            params = submit_args[1]
            tool_list = params["tools"]

            assert isinstance(tool_list, list)
            assert tool_list[0]["type"] == "function"
            assert tool_list[0]["name"] == "get_weather"
            assert "city" in tool_list[0]["parameters"]["properties"]

    def test_ask_llm_basic_call(self, mock_orchestrator):
        """Test basic ask_llm call"""
        agent = AgentContext("test_agent", mock_orchestrator)

        with agent:
            response = agent.ask_llm("Test prompt")

            assert isinstance(response, (ReadyLLMResponse, PendingLLMResponse))
            # Now backend.submit_query is called instead of provider.submit_query_to_provider
            mock_orchestrator._backend.submit_query.assert_called_once()

    def test_ask_llm_with_cache_hit(self, mock_orchestrator):
        """Test ask_llm when response is cached"""
        mock_orchestrator._backend.retrieve.return_value = ParsedResponse(
            text="Cached response",
            response_id="resp_123",
            metadata=None,
        )

        agent = AgentContext("test_agent", mock_orchestrator)

        with agent:
            response = agent.ask_llm("Cached prompt")

            assert isinstance(response, ReadyLLMResponse)
            assert response.final_answer == "Cached response"
            # Backend submit_query should not be called for cached responses
            mock_orchestrator._backend.submit_query.assert_not_called()

    def test_ask_llm_call_id_generation(self, mock_orchestrator):
        """Test that ask_llm generates correct call IDs"""
        agent = AgentContext("test_agent", mock_orchestrator)

        with agent:
            agent.ask_llm("Test prompt")

            # Verify backend was called with correct call_id structure
            call_args = mock_orchestrator._backend.submit_query.call_args
            call_id = call_args.kwargs["call_id"]

            assert call_id["agent_name"] == "test_agent"
            assert call_id["seq_id"] == 0  # First call
            assert call_id["session_id"] == 1
            assert call_id["meta"]["provider_type"] == "openai"

    def test_ask_llm_uses_llm_provider_type_in_call_metadata(self, mock_orchestrator):
        """Call metadata should track the selected llm provider for multiplexer compatibility."""
        mock_orchestrator._provider.provider_type = None
        agent = AgentContext("test_agent", mock_orchestrator)

        with agent:
            agent.ask_llm(
                "Test prompt",
                llm=LLMIdentity("gemini-2.5-flash", provider_type="google"),
            )

            call_args = mock_orchestrator._backend.submit_query.call_args
            call_id = call_args.kwargs["call_id"]
            assert call_id["meta"]["provider_type"] == "google"

    @patch("parallem.core.agent.agent.compute_hash")
    def test_ask_llm_hash_computation(self, mock_compute_hash, mock_orchestrator):
        """Test that ask_llm computes hashes correctly"""
        mock_compute_hash.return_value = "test_hash_123"

        agent = AgentContext("test_agent", mock_orchestrator)

        with agent:
            agent.ask_llm("Test prompt", instructions="Test instructions")

            mock_compute_hash.assert_called_once_with(
                "Test instructions", ["Test prompt"], salt=None
            )

    @patch("parallem.core.agent.agent.compute_hash")
    def test_ask_llm_hash_by_tools(self, mock_compute_hash, mock_orchestrator):
        """hash_by=['tools'] is currently unwired and should not alter salt."""
        mock_compute_hash.return_value = "test_hash_123"

        agent = AgentContext("test_agent", mock_orchestrator)
        tools = [{"type": "web_search", "extra": {"b": 2, "a": 1}}]

        with agent:
            agent.ask_llm(
                "Test prompt",
                instructions="Test instructions",
                hash_by=["tools"],
                tools=tools,
            )

        mock_compute_hash.assert_called_once_with(
            "Test instructions",
            ["Test prompt"],
            salt=None,
        )

    def test_context_manager_preserves_anonymous_counter(self, mock_orchestrator):
        """Test that context manager preserves anonymous counter across contexts"""
        agent = AgentContext("test_agent", mock_orchestrator)

        with agent:
            agent.ask_llm("first call")
            agent.ask_llm("second call")
            call_ids = [
                call.kwargs["call_id"]["seq_id"]
                for call in mock_orchestrator._backend.submit_query.call_args_list
            ]
            assert call_ids[-2:] == [0, 1]

        # Second context block - counter should continue
        mock_orchestrator._backend.submit_query.reset_mock()
        with agent:
            agent.ask_llm("third call")
            call_id = mock_orchestrator._backend.submit_query.call_args.kwargs["call_id"]
            assert call_id["seq_id"] == 2

    def test_ask_human_returns_human_response_and_persists(self, mock_orchestrator):
        """ask_human should persist with origin_type=1 and return HumanResponse."""
        agent = AgentContext("test_agent", mock_orchestrator)

        ds = mock_orchestrator._backend._get_datastore.return_value
        ds.retrieve.return_value = None
        with agent:
            response = agent.ask_human("Question?", [], input_fn=lambda _: "answer")

        assert isinstance(response, HumanResponse)
        assert response.final_answer == "answer"
        ds.store.assert_called_once()
        ds.retrieve.assert_called_once()
        assert ds.retrieve.call_args.kwargs["origin_type"] == 1
        call_args = ds.store.call_args
        assert call_args.kwargs["origin_type"] == 1

    def test_ask_human_increments_counter(self, mock_orchestrator):
        """ask_human should consume an anonymous sequence slot."""
        agent = AgentContext("test_agent", mock_orchestrator)

        ds = mock_orchestrator._backend._get_datastore.return_value
        ds.retrieve.return_value = None

        with agent:
            agent.ask_human("q1", [], input_fn=lambda _: "a1")
            agent.ask_human("q2", ["a1"], input_fn=lambda _: "a2")

        call_ids = [
            call.args[0]["seq_id"]
            for call in ds.store.call_args_list
        ]
        assert call_ids == [0, 1]

    @patch("parallem.core.agent.agent.compute_hash")
    def test_ask_human_hash_uses_prompt_and_documents(
        self, mock_compute_hash, mock_orchestrator
    ):
        """ask_human should hash using prompt as instructions and documents as basis."""
        mock_compute_hash.return_value = "human_hash_123"
        agent = AgentContext("test_agent", mock_orchestrator)

        ds = mock_orchestrator._backend._get_datastore.return_value
        ds.retrieve.return_value = None

        with agent:
            agent.ask_human(
                "System prompt",
                ("user", "hello"),
                ("assistant", "world"),
                salt="s1",
                input_fn=lambda _: "answer",
            )

        mock_compute_hash.assert_called_once_with(
            "System prompt",
            [("user", "hello"), ("assistant", "world")],
            salt="s1",
        )

    def test_ask_human_cache_hit_skips_prompt_and_store(self, mock_orchestrator):
        """ask_human should return cached human response when available."""
        agent = AgentContext("test_agent", mock_orchestrator)

        ds = mock_orchestrator._backend._get_datastore.return_value
        ds.retrieve.return_value = ParsedResponse(
            text="cached-human",
            response_id=None,
            metadata=None,
            old_seq_id=9,
            old_session_id=7,
        )

        def _boom(_):
            raise AssertionError("input_fn should not be called on cache hit")

        with agent:
            response = agent.ask_human("Question?", "foo", input_fn=_boom)

        assert isinstance(response, HumanResponse)
        assert response.final_answer == "cached-human"
        assert response.call_id["seq_id"] == 9
        assert response.call_id["session_id"] == 7
        ds.store.assert_not_called()

    def test_msg_state_ask_human_appends_human_response(self, mock_orchestrator):
        """MessageState.ask_human should append a HumanResponse."""
        agent = AgentContext("test_agent", mock_orchestrator)

        ds = mock_orchestrator._backend._get_datastore.return_value
        ds.retrieve.return_value = None

        with agent:
            msg_state = agent.get_msg_state()
            before_len = len(msg_state)
            response = msg_state.ask_human("Question?", input_fn=lambda _: "answer")

        assert isinstance(response, HumanResponse)
        assert response.final_answer == "answer"
        assert len(msg_state) == before_len + 1
        assert msg_state[-1] is response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
