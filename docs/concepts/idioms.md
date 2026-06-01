Code parallelization can take many forms. ParaLLeM supports these 3.

## Direct Idiom

The direct idiom is simplest to write. Declare agents in a `for` loop.

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

### Async strategy

You can sometimes achieve parallelization by combining the `async` strategy with synchronous code. Care must be taken to ensure that one agent does not block the other.

```python
import parallem as pllm
from dotenv import load_dotenv


def power_of_n_agent(agt: pllm.AgentContext, n: int):
    return agt.ask_llm(f"Name a power of {n}")


load_dotenv()
with pllm.resume_directory(
    ".pllm/simplest",
    strategy="async",
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

It is effective with `sync`, `async`, and `batch` strategies.

```python
--8<-- "examples/async/example_async.py"
```

!!! note
    `orch.run_agents` is similar to `asyncio.run(asyncio.gather(agts))`. However, `orch.run_agents` is recommended when using batch mode, because `orch.run_agents` properly handles ParaLLeM's interrupt semantics.
