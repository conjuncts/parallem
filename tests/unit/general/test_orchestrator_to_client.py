import asyncio
import inspect
from uuid import uuid4

import pytest

from parallem.core.gateway import resume_directory
from parallem.tools.server import WebSearchTool


class _FakeResponsesClient:
    def __init__(self, payloads):
        self.calls = []
        self._payloads = list(payloads)

    def set_payloads(self, payloads):
        self._payloads = list(payloads)

    def clear(self):
        self.calls.clear()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._payloads.pop(0)


class _FakeOpenAIClient:
    def __init__(self, payloads):
        self.responses = _FakeResponsesClient(payloads)


class _FakeAnthropicMessagesClient:
    def __init__(self, payloads):
        self.calls = []
        self._payloads = list(payloads)

    def set_payloads(self, payloads):
        self._payloads = list(payloads)

    def clear(self):
        self.calls.clear()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._payloads.pop(0)


class _FakeAnthropicClient:
    def __init__(self, payloads):
        self.messages = _FakeAnthropicMessagesClient(payloads)


# pytestmark = pytest.mark.skip(reason="slow")


@pytest.fixture(scope="session")
def fake_openai_orch(session_orch_root):
    orch_dir = session_orch_root / "sync-fake-openai"
    fake_client = _FakeOpenAIClient([])
    orch = resume_directory(
        orch_dir,
        provider="openai",
        strategy="sync",
        client=fake_client,
    )
    orch._fake_client = fake_client
    yield orch
    orch.finalize_and_persist()


@pytest.fixture(scope="session")
def fake_anthropic_orch(session_orch_root):
    orch_dir = session_orch_root / "sync-fake-anthropic"
    fake_client = _FakeAnthropicClient([])
    orch = resume_directory(
        orch_dir,
        provider="anthropic",
        strategy="sync",
        client=fake_client,
    )
    orch._fake_client = fake_client
    yield orch
    orch.finalize_and_persist()


@pytest.fixture
def test_agent_name(request):
    return f"{request.node.name}-{uuid4().hex}"


def test_to_client_sync_responses_create(shared_sync_orch, test_agent_name):
    mock_client = shared_sync_orch._mock_client
    mock_client.clear()
    mock_client.set_responses(["Hello from ask_llm"])

    client = shared_sync_orch.to_client(agent_name=test_agent_name)
    response = client.responses.create(
        model="gpt-5-nano",
        instructions="You are helpful",
        input=[{"role": "user", "content": "Say hello"}],
    )

    assert response.output_text == "Hello from ask_llm"
    assert response.output[-1]["type"] == "message"
    assert response.output[-1]["role"] == "assistant"
    assert response.output[-1]["content"][0]["text"] == "Hello from ask_llm"


def test_to_client_sync_responses_parse(shared_sync_orch, test_agent_name):
    mock_client = shared_sync_orch._mock_client
    mock_client.clear()
    mock_client.set_responses(['{"answer": 42}'])

    client = shared_sync_orch.to_client(agent_name=test_agent_name)
    response = client.responses.parse(
        model="gpt-5-nano",
        input="Return JSON",
        text_format={"type": "json_schema"},
    )

    assert response.output_text == '{"answer": 42}'
    assert response.output_parsed == {"answer": 42}


def test_to_client_sync_chat_completions_create(shared_sync_orch, test_agent_name):
    mock_client = shared_sync_orch._mock_client
    mock_client.clear()
    mock_client.set_responses(["Chat completion response"])

    client = shared_sync_orch.to_client(agent_name=test_agent_name)
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Summarize"}],
    )

    assert response.object == "chat.completion"
    assert response.choices[0]["message"]["content"] == "Chat completion response"


def test_to_client_concurrent_responses_create_is_async(
    shared_concurrent_orch, test_agent_name
):
    mock_client = shared_concurrent_orch._mock_client
    mock_client.clear()
    mock_client.set_responses(["Async hello"])

    client = shared_concurrent_orch.to_client(agent_name=test_agent_name)
    assert inspect.iscoroutinefunction(client.responses.create)

    response = asyncio.run(
        client.responses.create(
            model="gpt-5-nano",
            input=[{"role": "user", "content": "Hi"}],
        )
    )

    assert response.output_text == "Async hello"


def test_to_client_sync_caches_identical_requests(shared_sync_orch, test_agent_name):
    mock_client = shared_sync_orch._mock_client
    mock_client.clear()
    mock_client.set_responses(["First live response", "Second live response"])

    client = shared_sync_orch.to_client(agent_name=test_agent_name)

    response1 = client.responses.create(
        model="gpt-5-nano",
        instructions="Be concise",
        input=[{"role": "user", "content": "Cache me"}],
    )
    response2 = client.responses.create(
        model="gpt-5-nano",
        instructions="Be concise",
        input=[{"role": "user", "content": "Cache me"}],
    )

    assert response1.output_text == "First live response"
    assert response2.output_text == "First live response"
    assert len(mock_client.calls) == 1


def test_to_client_sync_hash_by_llm_differentiates_cache(
    shared_sync_orch, test_agent_name
):
    mock_client = shared_sync_orch._mock_client
    mock_client.clear()
    mock_client.set_responses(["nano response", "mini response"])

    client = shared_sync_orch.to_client(agent_name=test_agent_name)

    response1 = client.responses.create(
        model="gpt-5-nano",
        instructions="Be concise",
        input=[{"role": "user", "content": "Cache by model"}],
    )
    response2 = client.responses.create(
        model="gpt-5-mini",
        instructions="Be concise",
        input=[{"role": "user", "content": "Cache by model"}],
    )

    assert response1.output_text == "nano response"
    assert response2.output_text == "mini response"
    assert len(mock_client.calls) == 2


def test_to_client_sync_hash_by_tools_does_not_differentiate_cache_yet(
    shared_sync_orch, test_agent_name
):
    mock_client = shared_sync_orch._mock_client
    mock_client.clear()
    mock_client.set_responses(["toolset A response", "toolset B response"])

    client = shared_sync_orch.to_client(
        agent_name=test_agent_name,
        ask_params={"hash_by": ["llm", "tools"]},
    )

    response1 = client.responses.create(
        model="gpt-5-nano",
        instructions="Be concise",
        input=[{"role": "user", "content": "Cache by tools"}],
        tools=[{"type": "web_search"}],
    )
    response2 = client.responses.create(
        model="gpt-5-nano",
        instructions="Be concise",
        input=[{"role": "user", "content": "Cache by tools"}],
        tools=[{"type": "code_interpreter"}],
    )

    assert response1.output_text == "toolset A response"
    assert response2.output_text == "toolset A response"
    assert len(mock_client.calls) == 1


def test_to_client_sync_surfaces_function_calls(fake_openai_orch, test_agent_name):
    fake_payload = {
        "id": "resp_tool_1",
        "output": [
            {
                "id": "fc_12345xyz",
                "call_id": "call_123",
                "type": "function_call",
                "name": "lookup_weather",
                "arguments": '{"city": "Boston"}',
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I'll call the weather tool now.",
                    }
                ],
            },
        ],
    }
    fake_client = fake_openai_orch._fake_client
    fake_client.responses.clear()
    fake_client.responses.set_payloads([fake_payload])

    client = fake_openai_orch.to_client(agent_name=test_agent_name)
    response = client.responses.create(
        model="gpt-5-nano",
        input=[{"role": "user", "content": "What's weather in Boston?"}],
    )

    assert response.output_text == "I'll call the weather tool now."
    tool_calls = [item for item in response.output if item["type"] == "function_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "lookup_weather"
    assert tool_calls[0]["arguments"] == '{"city": "Boston"}'
    assert tool_calls[0]["call_id"] == "call_123"


def test_to_client_sync_forwards_function_call_output_input(
    fake_anthropic_orch, test_agent_name
):
    fake_client = fake_anthropic_orch._fake_client
    fake_client.messages.clear()
    fake_client.messages.set_payloads(
        [
            {
                "id": "resp_tool_out_1",
                "content": [
                    {
                        "type": "text",
                        "text": "Thanks!",
                    }
                ],
            }
        ]
    )

    client = fake_anthropic_orch.to_client(agent_name=test_agent_name)
    _ = client.responses.create(
        model="claude-haiku-4-5-20251001",
        input=[
            {
                "type": "function_call_output",
                "call_id": "call_abc",
                "name": "lookup_weather",
                "output": "72F and sunny",
            }
        ],
    )

    assert len(fake_client.messages.calls) == 1
    sent_messages = fake_client.messages.calls[0]["messages"]
    assert len(sent_messages) == 1
    assert sent_messages[0]["role"] == "user"
    assert sent_messages[0]["content"][0]["type"] == "tool_result"
    assert sent_messages[0]["content"][0]["tool_use_id"] == "call_abc"
    assert sent_messages[0]["content"][0]["content"] == "72F and sunny"


def test_to_client_sync_forwards_web_search_tool_to_openai(
    fake_openai_orch, test_agent_name
):
    fake_client = fake_openai_orch._fake_client
    fake_client.responses.clear()
    fake_client.responses.set_payloads(
        [
            {
                "id": "resp_ws_openai_1",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Found info"}],
                    }
                ],
            }
        ]
    )

    client = fake_openai_orch.to_client(agent_name=test_agent_name)
    _ = client.responses.create(
        model="gpt-5-nano",
        input=[{"role": "user", "content": "search recent weather"}],
        tools=[
            WebSearchTool(
                {
                    "user_location": {
                        "type": "approximate",
                        "country": "US",
                    }
                }
            )
        ],
    )

    assert len(fake_client.responses.calls) == 1
    sent_tools = fake_client.responses.calls[0]["tools"]
    assert len(sent_tools) == 1
    assert sent_tools[0]["type"] == "web_search"
    assert sent_tools[0]["user_location"]["country"] == "US"


def test_to_client_sync_forwards_web_search_tool_to_anthropic(
    fake_anthropic_orch, test_agent_name
):
    fake_client = fake_anthropic_orch._fake_client
    fake_client.messages.clear()
    fake_client.messages.set_payloads(
        [
            {
                "id": "resp_ws_anthropic_1",
                "content": [{"type": "text", "text": "Found info"}],
            }
        ]
    )

    client = fake_anthropic_orch.to_client(agent_name=test_agent_name)
    _ = client.responses.create(
        model="claude-haiku-4-5-20251001",
        input=[{"role": "user", "content": "search recent weather"}],
        tools=[WebSearchTool({"max_uses": 2})],
    )

    assert len(fake_client.messages.calls) == 1
    sent_tools = fake_client.messages.calls[0]["tools"]
    assert len(sent_tools) == 1
    assert sent_tools[0]["type"] == "web_search_20260209"
    assert sent_tools[0]["name"] == "web_search"
    assert sent_tools[0]["max_uses"] == 2
