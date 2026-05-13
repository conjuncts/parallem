# ParaLLeM

An expressive library for calling LLMs in bulk

- Batch API support (50% discount!)
- Workflows simply described in Python
- Concise, lightweight

## Quickstart

```bash
pip install parallem
```

```python
import parallem as pllm

# Place OPENAI_API_KEY in .env file
with pllm.resume_directory(
    ".pllm/simplest",
    provider="openai",
    strategy="sync",
    dashboard=True,
    load_dotenv=True,
) as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm("Please name a power of 3.")
        print(resp.final_answer)
```

To switch to the Batch API, simply change `strategy="sync"` to `strategy="batch"`.

Responses are saved and cached. On the subsequent runs, answers are instant.

## Compatibility

| Sync/Batch | OpenAI | Anthropic | Google |
| --- | --- | --- | --- |
| Simple | ✅ | ✅ | ✅ |
| Structured Output | ✅ | ✅ | ✅ |
| Function Calls | ✅ | ✅ | ✅ |
| Web Search | ✅ | ✅ | ✅ |
| Image Input | ✅ | ✅ | ✅ |
| MCP | ✅ | ✅ | - |


## Philosophy

1. Switch between Synchronous and Batch API in 1 line of code.
2. Control flow is best represented with Python, not data structures.
3. An agent is more than just one LLM.

## Documentation

Please refer to the [documentation](https://parallem-ai.github.io/parallem).

[parallem.org](https://parallem.org)
