# MessageState API

`MessageState` is simply a **list** of messages — strings, images, and LLM responses — that automatically tracks conversation history and reduces boilerplate when doing multi-turn or tool-use workflows.

## Getting a MessageState

```python
with orch.agent() as agt:
    msgs = agt.get_msg_state()
```

`MessageState` is scoped to an agent. Each agent has one persistent `MessageState`.

## Basic usage

Strings, images, and `LLMResponse` objects can all be appended directly. Passing a `MessageState` to `ask_llm` sends the entire conversation as context.

```python
--8<-- "examples/simplest_msg_state.py"
```

## With tool use

MessageState takes care of feeding prior responses back into `ask_llm`, making multi-step function-calling loops concise:

```python
--8<-- "examples/tools/msg_state_tool.py"
```

## Persistence — `save` and `load`

`MessageState` can be checkpointed to the session directory and restored on subsequent runs. This is the foundation for long-running, resumable pipelines.

```python
--8<-- "examples/checkpoint/msg_state_checkpoint_recipe.py"
```

## Key methods

| Method | Description |
|---|---|
| `ask_llm(...)` | Forward to the underlying agent. When called with no arguments, uses the current list as context. |
| `ask_functions(...)` | Invoke user functions for any pending function calls in the last response. |
| `save()` | Persist the current message list to disk. |
| `load()` | Load a previously saved message list from disk. |
| `get_state_hash(salt=None)` | Compute a deterministic hash of the current conversation state. |
