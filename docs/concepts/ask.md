# Ask API

An agent is a program that can ask not just LLMs, but also functions and humans. ParaLLeM unifies `ask_llm`, `ask_functions`, and `ask_human` into a common interface.

## `ask_llm`

Send a prompt (or full conversation) to an LLM and receive a lazy `LLMResponse`.

```python
resp = agt.ask_llm("What is the capital of France?")
print(resp.final_answer)
```

The response is lazy-loaded — it is not resolved until you call `.final_answer` or iterate over function calls. This lets the runtime batch and cache calls efficiently.

### Structured output

Pass a Pydantic model to `structured_output` to get back a validated object:

```python
from pydantic import BaseModel

class Answer(BaseModel):
    capital: str

resp = agt.ask_llm("What is the capital of France?", structured_output=Answer)
print(resp.final_json)  # {"capital":"Paris"}
```

### Tool use

```python title="examples/simplest_tool.py"
--8<-- "examples/simplest_tool.py"
```

Output:
```python
[
    'Add 3 and 4.',
    '',
    FunctionCallOutput(name=add, call_id=, content=7...),
    '3 + 4 = 7'
]
```

## `ask_functions`

After `ask_llm` returns a response that contains function calls, `ask_functions` executes them by running the actual Python functions.

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

    # Must pass the conversation for hashing purposes.
    # ParaLLeM must associate the human response with
    # the right point in the conversation.
    conv,
)
print(resp.final_answer)
```

It follows the same hashing idea as `ask_llm`:

- `prompt` acts like the system prompt/instructions.
- `documents` and `*additional_documents` are the hash basis.
- `salt` can be used to force differentiation.
