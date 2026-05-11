Code parallelization can take many forms. ParaLLeM supports these 3.

## Direct Idiom

The direct idiom (Direct API) is simplest to write. Declare agents in a `for` loop.

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

It is effective with `sync` and `batch`. Batch parallelization is effective because of ParaLLeM's interrupt semantics.

However, it is *not* effective if ran asynchronously, because power-of-2 agent must finish before power-of-3 agent can begin.

That is a limitation of python: you need `await`, `async`, and `asyncio.run` to allow async calls.

## Concurrent Idiom

You can use the `concurrent` strategy with the Direct API to achieve parallelization. However, you must ensure that one agent does not block the other. 

This idiom is less efficient than true async. Typically, `await` yields control between tasks, but here that is not possible.

However, you still achieve parallelization which resembles async execution. 

```python
import parallem as pllm
from dotenv import load_dotenv


def power_of_n_agent(agt: pllm.AgentContext, n: int):
    return agt.ask_llm(f"Name a power of {n}")


load_dotenv()
with pllm.resume_directory(
    ".pllm/simplest",
    strategy="concurrent",
) as orch:
    collector: list[pllm.LLMResponse] = []
    for i in range(2, 6):
        with orch.agent(i) as agt:
            collector.append(power_of_n_agent(agt, i))
    
    out = orch.resolve_all(collector)
    print(out)
```



## Async Idiom

If you have async functions, you can use the async idiom.

It is effective with `sync`, `concurrent`, and `batch` strategies.

```python
--8<-- "examples/async/simplest_async.py"
```

However, the async idiom is trickier to write and port to.

!!! note
    `orch.run_agents` is similar to `asyncio.run(asyncio.gather(agts))`. However, `orch.run_agents` is recommended when using batch mode, because `orch.run_agents` properly handles ParaLLeM's interrupt semantics.
