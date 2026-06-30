from uuid import uuid4

import pytest

from parallem.core.response import PendingLLMResponse
from parallem.testing.simple_backend import MockBackend
from parallem.types import ParsedResponse


@pytest.fixture
def test_agent_name(request):
    return f"{request.node.name}-{uuid4().hex}"


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
    response = PendingLLMResponse(call_id=generic_call_id, backer=backend)

    ready_response = await response
    assert ready_response.final_answer == "awaited-response"
    assert response.final_answer == "awaited-response"
