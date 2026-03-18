# pipelinellm

`pipelinellm` is a batch-first Python library for building LLM workflows without giving up ordinary Python control flow.

## Why use it?

- Switch between sync, concurrent, and batch execution with minimal code changes.
- Keep orchestration logic in Python instead of separate graph or DSL layers.
- Reuse the same workflow structure across tool calls, structured output, image input, and web search.
- Resume runs from disk and reuse cached responses for fast iteration.

## Installation

```bash
pip install pipelinellm
```

## Small example

```python
from dotenv import load_dotenv
import pipelinellm as pllm

load_dotenv()

with pllm.resume_directory(
    ".pllm/simplest",
    provider="openai",
    strategy="sync",
    dashboard=True,
) as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm("Please name a power of 3.", hash_by=["llm"])
        agt.print(resp.final_answer)
```

## Read next

- Start with the quickstart for the main workflow.
- Compare sync, concurrent, and batch in the strategy guide.
- Read the philosophy page for the design goals behind the library.