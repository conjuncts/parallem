
# Quickstart

A demonstration of supported features, including function calling, image input, structured output, and web search.

```python title="examples/full_tour.py"
--8<-- "examples/full_tour.py"
```


<div class="highlight session-log-html"><pre><code>[<span class="log-tag">INFO</span>] Resuming with session_id=0
1. 243 (which is 3^5).
2. Apple Inc. (AAPL) is currently trading at $257.46 per share (latest trade: 01:15:00 UTC on March 7, 2026).
3. {"final_answer":"The capital of France is Paris."}
4. These are horses — domestic equines. The photo shows two adult horses standing in a grassy field.
FunctionCall(name=count_files, call_id=call_Xau, args={'directory': '~/examples'})
5.
6. There are four files in ~/examples.
[<span class="log-tag">DASH</span>] <span class="log-hash">↘ 2b55f032 ↘ f7824348 ↘ c9f2fcc4 ↘ f13b65b2 ↘ bc7f1641 ↘ 2883bea6</span>
</code></pre></div>

If the program is rerun, the results are cached and available immediately.

### Batch mode

Use the Batch API in just one line of code:

```python
with pllm.resume_directory(
    ".pllm/example/batch",
    provider="openai",
    strategy="batch",  # Only change!!!
    # ...
) as orch:
    # ...
```

<div class="highlight session-log-html"><pre><code>[<span class="log-tag">INFO</span>] Resuming with session_id=0
Submit 1 batch (6 calls)? (y/n/preview): y
Sent batch: batch_69b4a26290008190a08e246922784ed8
[<span class="log-tag">DASH</span>] <span class="log-hash">⇈ 69b4a262</span>
</code></pre></div>

After a while, rerun the program. The batch will automatically be downloaded and handled.
<div class="highlight session-log-html"><pre><code>[<span class="log-tag">INFO</span>] Resuming with session_id=1
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


!!! warning

    ParaLLeM caches requests by hash. By default, hashes only consider **input documents** and **LLM name**. If only a non-hashed parameter changes (ie. reasoning level), there will be a hash collision. To avoid this, customize `hash_by` or compute a custom `salt`. See [persistence](concepts/persistence.md).


## Advanced Usage

- See the docs for:
    - The [**MessageState** guide](concepts/msg_state.md): a simple list that automatically appends documents and responses. Tracks long conversations and reduces boilerplate.
    - The [**Ask** guide](concepts/ask.md): `ask_llm`, `ask_functions`.
    - The [**memoize** guide](concepts/memoize.md): for caching expensive or non-deterministic blocks of code.

## Further examples

A suite of examples (a "cookbook") is available under `examples/*`.
