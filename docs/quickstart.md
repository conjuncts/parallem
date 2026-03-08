
# Quickstart 1

This quickstart is the simplest workflow there is. See `examples/simplest.py`

```python
from dotenv import load_dotenv
import pipelinellm as pllm

load_dotenv()  # Put your OPENAI_API_KEY in the .env file

with pllm.resume_directory(
    ".pllm/simplest",
    provider="openai",
    strategy="sync",
    dashboard=True,
    # ignore_cache=True,
) as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm("Please name a power of 3.", hash_by=["llm"])

        agt.print(resp.final_answer)  # 27 (which is 3^3).
```

Here is what's printed to console:
```md
[INFO] Resuming with session_id=27
27 (which is 3^3).
[DASH] ↘ b14ccd95
```

If you run the program again, the results are cached and available instantly.

# Quickstart 2

This quickstart gives a tour of most capabilities of pipelinellm. See `examples/full/example_tour.py`.

Here is what's printed to console.

```md
[DEBUG] Resuming directory
[DEBUG] Creating backend
[DEBUG] Creating provider
[DEBUG] Creating AgentOrchestrator
[INFO] Resuming with session_id=23
1. 243 (which is 3^5).
2. Apple Inc. (AAPL) is currently trading at $257.46 per share (latest trade: 01:15:00 UTC on March 7, 2026).
3. {"final_answer":"The capital of France is Paris."}
4. These are horses — domestic equines. The photo shows two adult horses standing in a grassy field.
[FunctionCall(name=count_files, call_id=call_Xau, args={'directory': '~/examples'})]  
5.
6. There are four files in ~/examples.
[DASH] ↘ 2b55f032 ↘ f7824348 ↘ c9f2fcc4 ↘ f13b65b2 ↘ bc7f1641 ↘ 2883bea6
```

Notice that if you run the program again, the results are cached and available instantly.

Finally, you could have done the entire workflow in batch mode simply by changing one line of code:

```python
with pllm.resume_directory(
    ".pllm/simplest",
    provider="openai",
    strategy="batch",  # Only change!
    # ...
) as orch:
    # ...
```

# Quickstart 3

The **MessageState abstraction** helps manage long conversation threads. MessageState is simply a **list** that automatically stores documents and responses as they get added. A good example is `examples/tools/msg_state_tool.py`.

# Further examples

A suite of examples (a "cookbook") is available under `examples/*`.

# Caveats

Pipelinellm saves progress by hashing. However, **not all config settings are hashed.** For instance, tool definitions are not hashed. So if available tools change, then the hashes are still considered identical, so the old cached value is still returned. This behavior can be tuned by setting the `hash_by` parameter.
