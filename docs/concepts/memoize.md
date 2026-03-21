# Memoize API

`memoize` lets you cache an entire block of code — including LLM calls and arbitrary Python side effects — so that subsequent runs skip execution and replay the recorded results instantly.

## When to use it

For example, `memoize` is well-suited for blocks that are:

- **Non-deterministic** (e.g. `random`, live API calls)  
- **Expensive** (long clock time, heavy computation)  

!!! Important
    Memoize only tracks changes to MessageState. It depends on the hash of MessageState to decide whether to reuse a previously stored value.

## Basic usage

```python
--8<-- "examples/advanced/simplest_memoize.py"
```

!!! warning
    Only changes to MessageState is saved. That means you will **not** have access to local variables within the `memoize()` block.

## `memoize` signature

```python
agt.memoize(salt=None)
```

| Parameter | Description |
|---|---|
| `salt` | Optional string to differentiate memoize entries that share the same conversation hash. |

## `begin()`

Must be called inside the `with` block before any memoized work. Calling `begin()` is what triggers the cache lookup.

```python
with agt.memoize() as mem:
    mem.begin()
    # ... expensive or non-deterministic work ...
```

!!! warning
    If you omit `mem.begin()`, memoization will not occur.


## How it works

1. `agt.memoize()` returns a context manager that starts watching `MessageState` (and non-message state) for mutations.  
2. If the conversation state (hash) at the point `mem.begin()` matches a previous run, the block is short-circuited and all recorded `MessageState` mutations are replayed. Otherwise, execution proceeds.
3. On first run, all `MessageState` mutations performed inside the `with` block are recorded to disk.


```
First run:   begin() → no cache → track operations → exit → save log
Later runs:  begin() → cache hit → replay log → skip rest of block
```

## Interaction with MessageState

`memoize` records every `MessageState` mutation (append, extend, set item, etc.) and replays them in order. This means that after replay, `msgs[-1]` holds the same `LLMResponse` as the original run — complete with cached call IDs that resolve from the datastore without making a new LLM request.
