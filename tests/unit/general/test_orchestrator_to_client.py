import asyncio
import inspect

import pytest

from parallem.core.gateway import resume_directory
from parallem.testing.simple_mock import mock_openai_client
from parallem.tools.server import WebSearchTool


class _FakeResponsesClient:
    def __init__(self, payloads):
        self.calls = []
        self._payloads = list(payloads)

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

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._payloads.pop(0)


class _FakeAnthropicClient:
    def __init__(self, payloads):
        self.messages = _FakeAnthropicMessagesClient(payloads)


pytestmark = pytest.mark.skip(reason="slow")


def test_to_client_sync_responses_create(tmp_path):
    orch_dir = tmp_path / "sync"
    mock_client = mock_openai_client(responses=["Hello from ask_llm"])
    orch = resume_directory(
        str(orch_dir), provider="openai", strategy="sync", client=mock_client
    )

    client = orch.to_client(agent_name="openai-client")
    response = client.responses.create(
        model="gpt-5-nano",
        instructions="You are helpful",
        input=[{"role": "user", "content": "Say hello"}],
    )

    assert response.output_text == "Hello from ask_llm"
    assert response.output[-1]["type"] == "message"
    assert response.output[-1]["role"] == "assistant"
    assert response.output[-1]["content"][0]["text"] == "Hello from ask_llm"


def test_to_client_sync_responses_parse(tmp_path):
    orch_dir = tmp_path / "sync-parse"
    mock_client = mock_openai_client(responses=['{"answer": 42}'])
    orch = resume_directory(
        str(orch_dir), provider="openai", strategy="sync", client=mock_client
    )

    client = orch.to_client(agent_name="openai-client")
    response = client.responses.parse(
        model="gpt-5-nano",
        input="Return JSON",
        text_format={"type": "json_schema"},
    )

    assert response.output_text == '{"answer": 42}'
    assert response.output_parsed == {"answer": 42}


def test_to_client_sync_chat_completions_create(tmp_path):
    orch_dir = tmp_path / "sync-chat"
    mock_client = mock_openai_client(responses=["Chat completion response"])
    orch = resume_directory(
        str(orch_dir), provider="openai", strategy="sync", client=mock_client
    )

    client = orch.to_client(agent_name="openai-client")
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Summarize"}],
    )

    assert response.object == "chat.completion"
    assert response.choices[0]["message"]["content"] == "Chat completion response"


def test_to_client_concurrent_responses_create_is_async(tmp_path):
    orch_dir = tmp_path / "concurrent"
    mock_client = mock_openai_client(responses=["Async hello"], concurrent=True)
    orch = resume_directory(
        str(orch_dir), provider="openai", strategy="concurrent", client=mock_client
    )

    client = orch.to_client(agent_name="openai-client")
    assert inspect.iscoroutinefunction(client.responses.create)

    response = asyncio.run(
        client.responses.create(
            model="gpt-5-nano",
            input=[{"role": "user", "content": "Hi"}],
        )
    )

    assert response.output_text == "Async hello"


def test_to_client_sync_caches_identical_requests(tmp_path):
    orch_dir = tmp_path / "sync-cache"
    mock_client = mock_openai_client(
        responses=["First live response", "Second live response"]
    )
    orch = resume_directory(
        str(orch_dir), provider="openai", strategy="sync", client=mock_client
    )

    client = orch.to_client(agent_name="openai-client")

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


def test_to_client_sync_hash_by_llm_differentiates_cache(tmp_path):
    orch_dir = tmp_path / "sync-cache-by-llm"
    mock_client = mock_openai_client(responses=["nano response", "mini response"])
    orch = resume_directory(
        str(orch_dir),
        provider="openai",
        strategy="sync",
        client=mock_client,
        hash_by=["llm"],
    )

    client = orch.to_client(agent_name="openai-client")

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


def test_to_client_sync_surfaces_function_calls(tmp_path):
    orch_dir = tmp_path / "sync-tools-output"

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
    fake_client = _FakeOpenAIClient([fake_payload])

    orch = resume_directory(
        str(orch_dir),
        provider="openai",
        strategy="sync",
        client=fake_client,
    )

    client = orch.to_client(agent_name="openai-client")
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


def test_to_client_sync_forwards_function_call_output_input(tmp_path):
    orch_dir = tmp_path / "sync-tools-input"

    fake_client = _FakeAnthropicClient(
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

    orch = resume_directory(
        str(orch_dir),
        provider="anthropic",
        strategy="sync",
        client=fake_client,
    )

    client = orch.to_client(agent_name="openai-client")
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


def test_to_client_sync_forwards_web_search_tool_to_openai(tmp_path):
    orch_dir = tmp_path / "sync-web-search-openai"

    fake_client = _FakeOpenAIClient(
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

    orch = resume_directory(
        str(orch_dir),
        provider="openai",
        strategy="sync",
        client=fake_client,
    )

    client = orch.to_client(agent_name="openai-client")
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


def test_to_client_sync_forwards_web_search_tool_to_anthropic(tmp_path):
    orch_dir = tmp_path / "sync-web-search-anthropic"

    fake_client = _FakeAnthropicClient(
        [
            {
                "id": "resp_ws_anthropic_1",
                "content": [{"type": "text", "text": "Found info"}],
            }
        ]
    )

    orch = resume_directory(
        str(orch_dir),
        provider="anthropic",
        strategy="sync",
        client=fake_client,
    )

    client = orch.to_client(agent_name="openai-client")
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
