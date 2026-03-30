import asyncio
import logging

import pytest

from parallem.core.agent.orchestrator import AgentOrchestrator
from parallem.core.file_manager import FileManager
from parallem.core.response import PendingLLMResponse
from parallem.logging.dash_logger import PrimitiveDashboardLogger
from parallem.testing.simple_backend import MockBackend
from parallem.types import ParsedResponse


def _make_orchestrator(tmp_path, *, strategy: str):
    fm = FileManager(str(tmp_path / "session"))
    provider = object()
    backend = MockBackend()
    return AgentOrchestrator(
        file_manager=fm,
        backend=backend,
        provider=provider,
        logger=logging.getLogger("parallem.test"),
        dashlog=PrimitiveDashboardLogger(),
        strategy=strategy,
    )


def test_run_agent_executes_async_fn_in_sync_mode(tmp_path):
    orchestrator = _make_orchestrator(tmp_path, strategy="sync")

    async def _agent_fn(agent):
        await asyncio.sleep(0)
        return f"done:{agent.agent_name}"

    result = orchestrator.run_agent(_agent_fn, agent_name="sync-agent")
    assert result == "done:sync-agent"


def test_run_agent_queues_and_persists_async_agents_in_concurrent_mode(tmp_path):
    orchestrator = _make_orchestrator(tmp_path, strategy="concurrent")
    seen = []

    async def _agent_fn(agent):
        await asyncio.sleep(0.01)
        seen.append(agent.agent_name)
        return agent.agent_name

    orchestrator.run_agent(_agent_fn, agent_name="a")
    orchestrator.run_agent(_agent_fn, agent_name="b")
    assert seen == []

    orchestrator.persist()
    assert sorted(seen) == ["a", "b"]


@pytest.mark.asyncio
async def test_llm_response_await_uses_backend_retrieve(generic_call_id):
    backend = MockBackend()
    backend.store(
        generic_call_id,
        ParsedResponse(
            text="awaited-response",
            response_id="r1",
            metadata=None,
        ),
    )
    response = PendingLLMResponse(call_id=generic_call_id, backend=backend)

    value = await response
    assert value == "awaited-response"
    assert response.final_answer == "awaited-response"
