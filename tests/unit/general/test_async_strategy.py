import asyncio
import logging

import pytest

from parallem.core.agent.orchestrator import AgentOrchestrator
from parallem.core.exception import NotAvailable
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

    result = orchestrator.create_agent(_agent_fn, agent_name="sync-agent")
    assert result.done()
    assert result.result() == "done:sync-agent"


def test_run_agent_queues_and_persists_async_agents_in_concurrent_mode(tmp_path):
    orchestrator = _make_orchestrator(tmp_path, strategy="concurrent")
    seen = []

    async def _agent_fn(agent):
        await asyncio.sleep(0.01)
        seen.append(agent.agent_name)
        return agent.agent_name

    fut1 = orchestrator.create_agent(_agent_fn, agent_name="a")
    fut2 = orchestrator.create_agent(_agent_fn, agent_name="b")
    assert not fut1.done()
    assert not fut2.done()
    assert seen == []

    out = orchestrator.run_agents(fut1, fut2)
    assert out == ["a", "b"]
    assert sorted(seen) == ["a", "b"]


def test_run_agents_raises_not_available_after_all_complete(tmp_path):
    orchestrator = _make_orchestrator(tmp_path, strategy="batch")
    seen = []

    async def _ok(agent):
        seen.append(agent.agent_name)
        return "ok"

    async def _na(agent):
        seen.append(agent.agent_name)
        raise NotAvailable()

    f1 = orchestrator.create_agent(_ok, agent_name="first")
    f2 = orchestrator.create_agent(_na, agent_name="second")

    with pytest.raises(NotAvailable):
        orchestrator.run_agents(f1, f2)

    assert sorted(seen) == ["first", "second"]


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
