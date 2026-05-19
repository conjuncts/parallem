# Persistence

ParaLLeM saves LLM responses locally by **hashing request content**. On subsequent runs, a matching hash returns the cached response instantly — no API call is made.

## How caching works

Every call to `ask_llm` computes a SHA-256 hash of:

- The system prompt (`instructions`)
- All input documents (strings, images, function call outputs, …)
- Any additional salt terms (see below)

If ParaLLeM has already seen your hash, then the previous value is returned immediately. Otherwise, a request is sent to the provider and stored.

```python
with pllm.resume_directory(
    ".pllm/myproject",
    provider="openai",
    strategy="sync"
) as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm("Name a prime number.")
        print(resp.final_answer)  # live on first run, instant on subsequent runs
```


## Hashing

### What is and isn't hashed

By default, only the **message content** is hashed — including `instructions` (system prompt) and all input documents. Config that is _not_ included in the hash:

- Model name / LLM identity
- Tool definitions
- Provider type

This means that if you change the model but keep the same prompt, the cached response from the old model is returned. Use `hash_by` or `salt` to avoid this.


### `hash_by`

`hash_by` is a list of named terms to fold into the hash. Available options are:

- `"llm"`: Include the LLM identity (model name/provider)
- `"tool_names"`: Include tool names only
- `"structured_output"`: Include structured output schema
- `"kwargs"`: Include extra kwargs passed to the request
- `"all"`: Include everything (equivalent to all of the above)

```python
agt.ask_llm("Name a prime.", hash_by=["llm"])
```

Now switching from `gpt-4o` to `gpt-4o-mini` produces a different hash and a separate cache entry.

```python
agt.ask_llm(
    "Search the web",
    tools=[{"type": "web_search"}],
    hash_by=["tool_names"]
)
```

Using `hash_by=["tool_names"]` ensures that different tool sets produce separate cache entries.


### `salt`

`salt` can distinguish otherwise identical content. Use it to bypass a cached result.

```python
agt.ask_llm("Name a prime.")
agt.ask_llm("Name a prime.", salt=1)  # Will not collide
```


## `ignore_cache` and `rewrite_cache`

| Parameter | Effect |
|---|---|
| `ignore_cache=True` | Always call the provider, ignoring any stored response. |
| `rewrite_cache=True` | Always call the provider and **overwrite** the stored response with the new one. |

Both are set on `resume_directory`:

```python
pllm.resume_directory(".pllm/myproject", provider="openai", ignore_cache=True)
```