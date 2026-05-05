# ParaLLeM

ParaLLeM is a library for orchestrating agentic LLM workflows.

- Batch API support
- Concise, readable, and **expressive**
- Developer-centered and lightweight
- Parallelize thousands of requests, while keeping reproducible traces for each run
- **Save 50% on all token costs**

Find out more about our mission — [parallem.org](https://parallem.org)

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
        agt.print(resp.final_answer)
```

To switch to the Batch API, simply change `strategy="sync"` to `strategy="batch"`.

## Compatibility

| Sync/Batch | OpenAI | Anthropic | Google |
| --- | --- | --- | --- |
| Simple | ✅ | ✅ | ✅ |
| Structured Output | ✅ | ✅ | ✅ |
| Function Calls | ✅ | ✅ | ✅ |
| Web Search | ✅ | ✅ | ✅ |
| Image Input | ✅ | ✅ | ✅ |
| MCP | ✅ | - | - |


## Philosophy

1. We are built from the ground up around the Batch API.
2. Switch between sync (sequential) and batch in just 1 line of code.
3. Control flow is best represented with Python, not data structures.
4. An agent is more than just one LLM.

## Documentation

Please refer to the [documentation](https://parallem-ai.github.io/parallem).
