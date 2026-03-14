# Persistence

Pipelinellm saves every LLM response to a local datastore keyed by a **hash** of the request content. On subsequent runs, a matching hash returns the cached response instantly — no API call is made.

## How caching works

Every call to `ask_llm` computes a SHA-256 hash of:

- The system prompt (`instructions`)
- All input documents (strings, images, function call outputs, …)
- Any additional salt terms (see below)

If pllm's internal datastore already contains a response for that hash, it is returned immediately. Otherwise the request is sent to the provider and the response is stored.

```python
with pllm.resume_directory(".pllm/myproject", provider="openai", strategy="sync") as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm("Name a prime number.")
        agt.print(resp.final_answer)  # live on first run, instant on subsequent runs
```

The datastore lives inside the directory passed to `resume_directory` (e.g. `.pllm/myproject/`).

## `ignore_cache` and `rewrite_cache`

| Parameter | Effect |
|---|---|
| `ignore_cache=True` | Always call the provider, ignoring any stored response. |
| `rewrite_cache=True` | Always call the provider and **overwrite** the stored response with the new one. |

Both are set on `resume_directory`:

```python
pllm.resume_directory(".pllm/myproject", provider="openai", ignore_cache=True)
```

## Hashing

### What is and isn't hashed

By default, only the **message content** is hashed — including `instructions` (system prompt) and all input documents. Config that is _not_ included in the hash:

- Model name / LLM identity
- Tool definitions
- Provider type

This means that if you change the model but keep the same prompt, the cached response from the old model is returned. Use `hash_by` or `salt` to avoid this.

### `hash_by`

`hash_by` is a list of named terms to fold into the hash. Currently the only supported value is `"llm"`, which appends the model identity string before hashing.

```python
# Set globally for all calls in the session
pllm.resume_directory(".pllm/myproject", provider="openai", hash_by=["llm"])

# Or per-call
agt.ask_llm("Name a prime.", hash_by=["llm"])
```

Now switching from `gpt-4o` to `gpt-4o-mini` produces a different hash and a separate cache entry.

### `salt`

`salt` is a free-form string you can use to distinguish otherwise identical content.

```python
agt.ask_llm("Name a prime.", salt="experiment-v2")
```

Use `salt` when you want to force a fresh response without clearing the whole cache — for example when tool definitions change (which are not hashed):

```python
agt.ask_llm(prompt, tools=my_tools, salt="tools-v2")
```

### Hierarchy

`salt` and `hash_by` can be set at three levels, with more specific values taking precedence:

1. `resume_directory(hash_by=...)` — session-wide default
2. `agt.ask_llm(hash_by=..., salt=...)` — per-call override
3. `msgs.ask_llm(hash_by=..., salt=...)` — per-call override via MessageState
