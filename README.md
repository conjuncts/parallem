# ParaLLeM

ParaLLeM is a library for orchestrating agentic LLM workflows.

We support the Batch API.

We're concise, readable, and expressive.

We're developer-centered and lightweight.

We help you parallelize thousands of requests, while keeping reproducible traces for each run.

We help you save 50% on all token costs.

Find out more about our mission -- [parallem.org](https://parallem.org)

## Quickstart

```bash
pip install parallem
```

```python
from dotenv import load_dotenv
import parallem as pllm

load_dotenv()  # Put OPENAI_API_KEY in the .env file

with pllm.resume_directory(
    ".pllm/simplest",
    provider="openai",
    strategy="sync",
    dashboard=True,
) as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm("Please name a power of 3.")
        agt.print(resp.final_answer)
```

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
