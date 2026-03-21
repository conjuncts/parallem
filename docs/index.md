# pipelinellm

`pipelinellm` is a general-purpose, batch-first Python library for building LLM workflows in ordinary Python control flow.

## Why use it?

- Switch between sync, concurrent, and batch execution with just 1 line of code.
- Write applications in Python instead of a data structure or domain-specific language.
- Supports agentic features like tool calls, structured output, image input, and web search.
- Saves responses to disk for fast iteration.

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

<div class="highlight session-log-html"><pre><code>[<span class="log-tag">INFO</span>] Resuming with session_id=0
27 (which is 3^3).
[<span class="log-tag">DASH</span>] <span class="log-hash">↘ b14ccd95</span>
</code></pre></div>

## Read next

- Start with the quickstart for the main workflow.
- Compare sync, concurrent, and batch in the strategy guide.
- Read the philosophy page for the design goals behind the library.