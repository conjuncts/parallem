# Ask API

We consider agents to be programs that can ask not just LLMs, but also functions and humans. ParaLLeM unifies `ask_llm`, `ask_functions`, and `ask_human` into a common interface.

## `ask_llm`

Send a prompt (or full conversation) to an LLM.

```python
resp = agt.ask_llm("What is the capital of France?")
print(resp.final_answer)
```

`ask_llm` takes a list of documents. It supports the following types:

- LLMDocument
    - `str`
    - `Tuple[Literal["user", "assistant", "system", "developer"], str]`
    - `PIL.Image.Image`
    - `pllm.FunctionCallRequest`
    - `pllm.FunctionCallOutput`
    - `pllm.MCPOutput`
- LLMResponse

`ask_llm` has these keyword arguments:

| Parameter | Type | Purpose |
|---|---|---|
| `instructions` | `str` | System prompt |
| `llm` | `pllm.LLMIdentity` | LLM to use |
| `salt` | `Any` | Hash differentiation value |
| `hash_by` | `list[str]` | Additional fields to include in hashing (e.g., `"llm"`) |
| `structured_output` | `pydantic.BaseModel` | Pydantic model or schema for validated output |
| `tools` | `list[dict]` | Functions or server-defined tools available to the LLM |
| `tag` | `str` | Optional metadata tag for the request |
| `save_input` | `bool` | Whether to persist input documents (default: `None`) |


!!! warning

    ParaLLeM caches requests by hash. By default, hashes only consider **input documents** and **LLM name**. If only a non-hashed parameter changes (ie. reasoning level), there will be a hash collision. To avoid this, customize `hash_by` or compute a custom `salt`. See [persistence](persistence.md).

### Structured output

Pass a Pydantic model to `structured_output` to get back a validated object:

```python
from pydantic import BaseModel

class Answer(BaseModel):
    capital: str

resp = agt.ask_llm("What is the capital of France?", structured_output=Answer)
print(resp.final_json)  # {"capital":"Paris"}
```


## `ask_functions`

If `ask_llm` returns a response that contains function calls, `ask_functions` executes them by running the actual Python functions.

```python title="Quickstart"
def count_files(directory: str) -> int:
    """Counts files in a directory"""
    return 4

# ...
resp = agt.ask_llm(prompt, tools=[count_files])
fc_outs = agt.ask_functions(resp, count_files=count_files)
final = agt.ask_llm([prompt, resp, *fc_outs])
```

## `ask_human`

`ask_human` is for human-in-the-loop checkpoints, approvals, or missing context that should come from a person instead of a model.

```python
conv = agt.get_msg_state()
resp = agt.ask_human(
    "Please confirm next step",
    conv,  # Must pass the conversation for hashing purposes.
)
print(resp.final_answer)
```

It follows the same hashing idea as `ask_llm`:

- `prompt` acts like the system prompt/instructions.
- `documents` and `*additional_documents` are the hash basis.
- `salt` can be used to force differentiation.
