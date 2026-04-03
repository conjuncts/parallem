# Migration Guide

## From OpenAI

`orch.to_client()` is an OpenAI-compatible facade designed for low-friction migration.

### Basic usage

```python
from dotenv import load_dotenv

from parallem.core.gateway import resume_directory

load_dotenv()


with resume_directory(".pllm/example/openai", provider="openai", strategy="sync") as orch:
    client = orch.to_client()
    resp = client.responses.create(
        model="gpt-5-nano",
        input=[{"role": "user", "content": "Hello"}],
    )
    print(resp.output_text)
```

```python
from parallem.core.gateway import resume_directory

weather_tool = {
    "type": "function",
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}

with resume_directory(".pllm/example/openai", provider="openai", strategy="sync") as orch:
    client = orch.to_client()

    resp = client.responses.create(
        model="gpt-5-nano",
        input=[{"role": "user", "content": "Hello"}],
    )
    print(resp.output_text)

    resp = client.responses.create(
        model="gpt-5-nano",
        input=[{"role": "user", "content": "What's the weather in Boston?"}],
        tools=[weather_tool],
    )

    tool_calls = [item for item in resp.output if item.get("type") == "function_call"]
    print(tool_calls)
```

### Supported APIs

- `responses.create`
- `responses.parse`
- `chat.completions.create`

### Supported behavior

- OpenAI-style request shape (`model`, `instructions`, `input`, `tools`, `text_format`)
- OpenAI-style response shape (`output_text`, `output`, `choices`)
- `function_call` and items surfaced in response output
- `function_call_output` items accepted as input
- Strategy-aware sync/async behavior:
  - `strategy="sync"`: synchronous methods
  - `strategy="concurrent"`: async methods

### Caching and hashing

- Uses the same hashing/caching pipeline as `ask_llm(...)`
- Identical effective requests reuse cache
- Hash differentiation options like `hash_by=["llm"]` behave the same as direct `ask_llm(...)`

### Tool support

- Forwards `tools=` through the provider tool conversion layer
- Supports provider-specific server tools supported by `parallem` (for example web search where available)
- Tool-result roundtrips are supported via `function_call_output` input

### Scope notes

- This facade intentionally targets the migration-critical OpenAI surfaces listed above
- Less common OpenAI SDK surfaces are not guaranteed to be implemented