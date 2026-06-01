from unittest.mock import Mock

import pytest

import parallem as pllm
from parallem.core.agent.agent import AgentContext
from parallem.types import FunctionCall, LLMResponse


class TestToToolSchemaAgentContext:
    def test_omits_agent_context_args_from_tool_schema(self):
        def tool_fn(value: int, ctx: "pllm.AgentContext", label: str = "x"):
            return f"{value}:{label}:{ctx.agent_name}"

        schema = pllm.to_tool_schema([tool_fn])[0]
        params = schema["parameters"]

        assert "value" in params["properties"]
        assert "label" in params["properties"]
        assert "ctx" not in params["properties"]
        assert params["required"] == ["value"]


class TestAskFunctionsAgentContextInjection:
    def test_injects_distinct_subagents_per_call(self, mock_orchestrator):
        agent = AgentContext("root", mock_orchestrator)
        response = LLMResponse("", call_id={"doc_hash": "h"})
        response._pr = Mock(
            function_calls=[
                FunctionCall("needs_ctx", {"value": 7}, call_id="cid-1"),
                FunctionCall("needs_ctx", {"value": 9}, call_id="cid-2"),
            ]
        )

        captured = []

        def needs_ctx(value: int, ctx: pllm.AgentContext):
            captured.append((value, ctx.agent_name))
            return f"ok:{ctx.agent_name}:{value}"

        outputs = agent.ask_functions(
            response,
            needs_ctx=needs_ctx,
            subagent_names=["worker_alpha", "worker_beta"],
        )

        assert outputs[0].content == "ok:worker_alpha:7"
        assert outputs[1].content == "ok:worker_beta:9"
        assert captured == [(7, "worker_alpha"), (9, "worker_beta")]

    def test_raises_when_injection_required_without_subagent_names(self, mock_orchestrator):
        agent = AgentContext("root", mock_orchestrator)
        response = LLMResponse("", call_id={"doc_hash": "h"})
        response._pr = Mock(
            function_calls=[FunctionCall("needs_ctx", {"value": 7}, call_id="cid-1")]
        )

        def needs_ctx(value: int, ctx: pllm.AgentContext):
            return value

        with pytest.raises(ValueError, match="requires AgentContext injection"):
            agent.ask_functions(response, needs_ctx=needs_ctx)

    def test_raises_when_not_enough_subagent_names(self, mock_orchestrator):
        agent = AgentContext("root", mock_orchestrator)
        response = LLMResponse("", call_id={"doc_hash": "h"})
        response._pr = Mock(
            function_calls=[
                FunctionCall("needs_ctx", {"value": 1}, call_id="cid-1"),
                FunctionCall("needs_ctx", {"value": 2}, call_id="cid-2"),
            ]
        )

        def needs_ctx(value: int, ctx: pllm.AgentContext):
            return value

        with pytest.raises(ValueError, match="Not enough subagent_names"):
            agent.ask_functions(
                response,
                needs_ctx=needs_ctx,
                subagent_names=["only-one"],
            )
