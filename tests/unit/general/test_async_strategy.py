import asyncio
from uuid import uuid4

import pytest

from parallem.core.exception import NotAvailable
from parallem.core.response import PendingLLMResponse
from parallem.testing.simple_backend import MockBackend
from parallem.types import ParsedResponse


@pytest.fixture
def test_agent_name(request):
    return f"{request.node.name}-{uuid4().hex}"


def test_run_agent_executes_async_fn_in_sync_mode(async_sync_orch, test_agent_name):
    async def _agent_fn(agent):
        await asyncio.sleep(0)
        return f"done:{agent.agent_name}"

    result = async_sync_orch.create_agent(_agent_fn, agent_name=test_agent_name)
    assert result.done()
    assert result.result() == f"done:{test_agent_name}"


def test_run_agent_queues_and_persists_async_agents_in_async_mode(
    async_async_orch, test_agent_name
):
    seen = []
    agent_name_a = f"{test_agent_name}-a"
    agent_name_b = f"{test_agent_name}-b"

    async def _agent_fn(agent):
        await asyncio.sleep(0.01)
        seen.append(agent.agent_name)
        return agent.agent_name

    fut1 = async_async_orch.create_agent(_agent_fn, agent_name=agent_name_a)
    fut2 = async_async_orch.create_agent(_agent_fn, agent_name=agent_name_b)
    assert not fut1.done()
    assert not fut2.done()
    assert seen == []

    out = async_async_orch.run_agents(fut1, fut2)
    assert out == [agent_name_a, agent_name_b]
    assert sorted(seen) == sorted([agent_name_a, agent_name_b])


def test_run_agents_raises_not_available_after_all_complete(async_batch_orch, test_agent_name):
    seen = []
    first_agent_name = f"{test_agent_name}-first"
    second_agent_name = f"{test_agent_name}-second"

    async def _ok(agent):
        seen.append(agent.agent_name)
        return "ok"

    async def _na(agent):
        seen.append(agent.agent_name)
        raise NotAvailable()

    f1 = async_batch_orch.create_agent(_ok, agent_name=first_agent_name)
    f2 = async_batch_orch.create_agent(_na, agent_name=second_agent_name)

    with pytest.raises(NotAvailable):
        async_batch_orch.run_agents(f1, f2)

    assert sorted(seen) == sorted([first_agent_name, second_agent_name])


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

    ready_response = await response
    assert ready_response.final_answer == "awaited-response"
    assert response.final_answer == "awaited-response"
