# Persistence

ParaLLeM saves LLM responses locally by **hashing request content**. On subsequent runs, a matching hash returns the cached response — no API call is made.

Every call to `ask_llm` computes a SHA-256 hash of:

- The system prompt (`instructions`)
- All input documents (strings, images, function call outputs, …)
- LLM name (ie. "gpt-5-nano")
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


## What is and isn't hashed

By default, only **message content** and **LLM name** are hashed. Config settings that are _not_ hashed:

- Tool definitions
- Structured output
- Provider-specific keyword arguments (ie. `reasoning_level`)

If there is a config change but the same prompt is used, there could be a hash collision. To avoid this, customize `hash_by` or compute a custom `salt`.

### `hash_by`

`hash_by` is a list of named terms to fold into the hash. By default, `hash_by=["llm"]`. Available options are:

- `"llm"`: Include the LLM identity (model name/provider)
- `"tool_names"`: Include tool names only
- `"structured_output"`: Include structured output schema
- `"kwargs"`: Include extra kwargs passed to the request
- `"all"`: Include everything (equivalent to all of the above)

Message content, like instructions and documents, are always hashed. 

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
