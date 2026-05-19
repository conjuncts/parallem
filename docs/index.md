# ParaLLeM

Agents + Batch API.

## Why use it?

- Batch API support (50% discount!)
- Expressive. Agents are [simple Python functions](https://github.com/parallem-ai/parallem/blob/main-prototype/examples/simplest_agent.py).
- Concise, lightweight.
- Durable by default.
- Built for [>1 million parallel requests](use-cases/stress_1m.md).

## Installation

```bash
pip install parallem
```

## Small example

```python
from dotenv import load_dotenv
import parallem as pllm

load_dotenv()

with pllm.resume_directory(
    ".pllm/simplest",
    provider="openai",
    strategy="sync",
    dashboard=True,
) as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm("Please name a power of 3.")
        print(resp.final_answer)
```

<div class="highlight session-log-html"><pre><code>[<span class="log-tag">INFO</span>] Resuming with session_id=0
27 (which is 3^3).
[<span class="log-tag">DASH</span>] <span class="log-hash">↘ b14ccd95</span>
</code></pre></div>

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

- See the [quickstart](quickstart.md) for more information.
- See the [examples](https://github.com/parallem-ai/parallem/tree/main-prototype/examples).
