
## Quickstart 1

This quickstart is the simplest workflow there is. See `examples/simplest.py`

```python
--8<-- "examples/simplest.py"
```

Here is what's printed to console:
<div class="highlight session-log-html"><pre><code>[<span class="log-tag">INFO</span>] Resuming with session_id=27
27 (which is 3^3).
[<span class="log-tag">DASH</span>] <span class="log-hash">↘ b14ccd95</span>
</code></pre></div>

If you run the program again, the results are cached and available instantly.

## Quickstart 2

This quickstart gives a tour of most capabilities of pipelinellm. See `examples/full/example_tour.py`.

```python
--8<-- "examples/full/example_tour.py"
```

Here is what's printed to console.

<div class="highlight session-log-html"><pre><code>[<span class="log-tag">DEBUG</span>] Resuming directory
[<span class="log-tag">DEBUG</span>] Creating backend
[<span class="log-tag">DEBUG</span>] Creating provider
[<span class="log-tag">DEBUG</span>] Creating AgentOrchestrator
[<span class="log-tag">INFO</span>] Resuming with session_id=0
1. 243 (which is 3^5).
2. Apple Inc. (AAPL) is currently trading at $257.46 per share (latest trade: 01:15:00 UTC on March 7, 2026).
3. {"final_answer":"The capital of France is Paris."}
4. These are horses — domestic equines. The photo shows two adult horses standing in a grassy field.
FunctionCall(name=count_files, call_id=call_Xau, args={'directory': '~/examples'})
5.
6. There are four files in ~/examples.
[<span class="log-tag">DASH</span>] <span class="log-hash">↘ 2b55f032 ↘ f7824348 ↘ c9f2fcc4 ↘ f13b65b2 ↘ bc7f1641 ↘ 2883bea6</span>
</code></pre></div>

If you run the program again, the results are cached and available instantly.

### Batch mode

The entire workflow can be done in batch mode simply by changing one line of code:

```python
with pllm.resume_directory(
    ".pllm/example/batch",
    provider="openai",
    strategy="batch",  # Only change!!!
    log_level=logging.DEBUG,
    dashboard=True,
    llm="gpt-5-mini-2025-08-07",
    hash_by=["llm"],
    ignore_cache=True,
) as orch:
    # ...
```

Here is what's printed to console.
<div class="highlight session-log-html"><pre><code>[<span class="log-tag">INFO</span>] Resuming with session_id=32
[<span class="log-tag">DEBUG</span>] Resuming directory
[<span class="log-tag">DEBUG</span>] Creating backend
[<span class="log-tag">DEBUG</span>] Creating provider
[<span class="log-tag">DEBUG</span>] Creating AgentOrchestrator
[<span class="log-tag">INFO</span>] Resuming with session_id=33
Submit 1 batch (6 calls)? (y/n/preview): y
Sent batch: batch_69b4a26290008190a08e246922784ed8
[<span class="log-tag">DASH</span>] <span class="log-hash">⇈ 69b4a262</span>
</code></pre></div>

After waiting a while, simply run the program again. The batch will automatically be downloaded and handled for you.
<div class="highlight session-log-html"><pre><code>[<span class="log-tag">INFO</span>] Resuming with session_id=34
[<span class="log-tag">DEBUG</span>] Resuming directory
[<span class="log-tag">DEBUG</span>] Creating backend
[<span class="log-tag">DEBUG</span>] Creating provider
[<span class="log-tag">DEBUG</span>] Creating AgentOrchestrator
[<span class="log-tag">INFO</span>] Resuming with session_id=35
Batch batch_69b4a26290008190a08e246922784ed8 completed and stored.
1. 243 (which is 3^5).
2. Apple Inc. (AAPL) is currently trading at $257.46 per share (latest trade: 01:15:00 UTC on March 7, 2026).
3. {"final_answer":"The capital of France is Paris."}
4. These are horses — domestic equines. The photo shows two adult horses standing in a grassy field.
FunctionCall(name=count_files, call_id=call_Xau, args={'directory': '~/examples'})
5.
6. There are four files in ~/examples.
<span class="log-hash">C 2b55f032 C f7824348 C c9f2fcc4 C f13b65b2 C bc7f1641 C 2883bea6</span>
</code></pre></div>

## Advanced Usage

- See the docs for:
    - The **MessageState** API: simply a **list** that automatically stores documents and responses as they get added. A good example is `examples/tools/msg_state_tool.py`. Helps track long conversations, reducing boilerplate.
    - The **Ask** API: `ask_llm`, `ask_functions` (invoking user functions), and `ask_human` (human-in-the-loop).
    - The **memoize** API: for storing expensive or non-deterministic function results.

## Further examples

A suite of examples (a "cookbook") is available under `examples/*`.

## Caveats

Pipelinellm saves progress by hashing. However, **not all config settings are hashed.** For instance, tool definitions are not hashed. So if available tools change, then the hashes are still considered identical, so the old cached value is still returned. This behavior can be tuned by setting the `hash_by` parameter. You can force avoid hash collisions by passing a `salt` parameter.

