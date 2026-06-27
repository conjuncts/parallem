from unittest.mock import Mock


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
                FunctionCall("needs_ctx", {"value": 7}, fcall_id="cid-1"),
                FunctionCall("needs_ctx", {"value": 9}, fcall_id="cid-2"),
            ]
        )

        captured = []

        def needs_ctx(value: int, ctx: pllm.AgentContext):
            captured.append((value, ctx.agent_name))
            return f"ok:{ctx.agent_name}:{value}"

        outputs = agent.ask_functions(
            response,
            needs_ctx=needs_ctx,
        )

        assert outputs[0].content == "ok:root/needs_ctx_0:7"
        assert outputs[1].content == "ok:root/needs_ctx_1:9"
        assert captured == [(7, "root/needs_ctx_0"), (9, "root/needs_ctx_1")]
