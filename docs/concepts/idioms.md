Parallelization takes many forms.

## Simple

Declare agents in a `for` loop.

```python
import parallem as pllm
from dotenv import load_dotenv


def power_of_n_agent(agt: pllm.AgentContext, n: int):
    return agt.ask_llm(f"Name a power of {n}").final_answer


load_dotenv()
with pllm.resume_directory(
    ".pllm/simplest",
    strategy="sync",  # or batch
) as orch:
    for i in range(2, 6):
        with orch.agent(i) as agt:
            print(power_of_n_agent(agt, i))
```

It is effective with `sync` and `batch`. However, it is *not* effective if ran asynchronously, because power-of-2 agent must finish before power-of-3 agent can begin.


## Async


It is effective with `sync`, `async`, and `batch` strategies.

```python
--8<-- "examples/async/example_async.py"
```

!!! note
    In the above example, an alternative is `asyncio.gather(*coros)`. But in batch mode, if the first child raises an error due to a missing value, that would prevent the remaining coroutines from being batched.

    Therefore, either `asyncio.gather(*coros, return_exceptions=True)` or `orch.gather(*coros)` are recommended because it gracefully handles if a value is not available.


In the current version of ParaLLeM, `strategy=async` creates a separate event loop, and `ask_llm` eagerly kicks off that task. `await conv.ask_llm` ensures that that task is complete.