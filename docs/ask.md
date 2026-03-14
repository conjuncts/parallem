# Ask API

The Ask API is the primary way to interact with LLMs, user functions, and humans. It is exposed on both `AgentContext` (`agt`) and `MessageState`.

## `ask_llm`

Send a prompt (or full conversation) to an LLM and receive a lazy `LLMResponse`.

```python
resp = agt.ask_llm("What is the capital of France?")
print(resp.final_answer)
```

The response is **lazy-loaded** — it is not resolved until you call `.final_answer`, `.resolve()`, or iterate over function calls. This lets the runtime batch and cache calls efficiently.

### Signature

```python
ask_llm(
    documents,            # str | image | LLMResponse | MessageState | list thereof
    *additional_documents,
    instructions=None,    # system prompt
    llm=None,             # override model, e.g. "gpt-4o"
    salt=None,            # manual hash differentiator
    hash_by=None,         # e.g. ["llm"] to include model name in hash
    text_format=None,     # Pydantic model for structured output
    tools=None,           # list of tool dicts or ServerTool objects
    tag=None,             # optional label for dashboards
    save_input=None,      # whether to persist input documents
)
```

### Structured output

Pass a Pydantic model to `text_format` to get back a validated object:

```python
from pydantic import BaseModel

class Answer(BaseModel):
    capital: str

resp = agt.ask_llm("What is the capital of France?", text_format=Answer)
print(resp.final_answer)  # {"capital":"Paris"}
```

### Tool use

```python
--8<-- "examples/simplest_tool.py"
```

## `ask_functions`

After `ask_llm` returns a response that contains function calls, `ask_functions` executes them by dispatching to the matching Python callables.

Take this example from [Quickstart 2](quickstart.md#quickstart-2):
```python
def count_files(directory: str) -> int:
    """Counts files in a directory"""
    return 4

# ...
resp = agt.ask_llm(prompt, tools=pllm.to_tool_schema([count_files]))
fc_outs = agt.ask_functions(resp, count_files=count_files)
final = agt.ask_llm([prompt, resp, *fc_outs])
```

Pass functions as keyword arguments (name → callable).

## Cross-provider example

Both `ask_llm` calls below use different models from different providers within the same session:

```python
--8<-- "examples/simplest_multi.py"
```
