# MessageState API

`MessageState` is simply a **list** of messages — strings, images, and LLM responses — that automatically tracks conversation history and reduces boilerplate when doing multi-turn or tool-use workflows.

## Basic usage

Let's say that you're having a long conversation. Using `agt.ask_llm`, you would have to keep track of messages manually:

```python
def long_conversation_agent(agt: pllm.AgentContext):
    msgs = []

    msgs.append("Who discovered gravity?")
    resp1 = agt.ask_llm(msgs)
    msgs.append(resp1)
    msgs.append("Tell me more!")
    resp2 = agt.ask_llm(msgs)
    msgs.append(resp2)
    msgs.append("What else did that person do?")
    resp3 = agt.ask_llm(msgs)
    msgs.append(resp3)
    msgs.append("Can you find me a biography?")
    resp4 = agt.ask_llm(msgs, tools=[pllm.tools.WebSearchTool()])
    msgs.append(resp4)
    print(resp4)
```

It can be cumbersome and repetitive to keep track of long conversations. To address this, pipelinellm has the concept of `MessageState`. Instead of asking on `agt`, ask directly on `MessageState`. Documents/responses will automatically be tracked and appended.

```python
def long_conversation_agent(agt: pllm.AgentContext):
    conv = agt.get_msg_state()
    conv.ask_llm("Who discovered gravity?")
    conv.ask_llm("Tell me more!")
    conv.ask_llm("What else did that person do?")
    conv.ask_llm("Can you find me a biography?", tools=[pllm.tools.WebSearchTool()])
    print(conv[-1])
```

`conv` is simply a list, so you can do list-y things to it:
```python
print(conv[0:2])
conv.append("What year was that person born?")
conv.pop(0)
```

MessageState supports any of the following types:

- LLMDocument, which is defined as one of:
    - `str`
    - `PIL.Image.Image`
    - `Tuple[Literal["user", "assistant", "system", "developer"], str]`
    - `FunctionCallRequest`
    - `FunctionCallOutput`
- LLMResponse

## With tool use

MessageState takes care of feeding prior responses back into `ask_llm`, making multi-step function-calling loops concise:

```python
def add(a: int, b: int) -> int:
    """Calculates a plus b."""

def multiply(a: int, b: int) -> int:
    """Calculates a times b."""
    return a * b

def calculation_agent(agt: pllm.AgentContext):
    conv = agt.get_msg_state()
    last_msg = conv.ask_llm(
        "Add 3 and 4.",
        tools=pllm.to_tool_schema([multiply, add]),
    )

    conv.ask_functions(add=add, multiply=multiply)
    last_msg = conv.ask_llm()
    agt.print(conv.resolve())
```

## Persistence — `save` and `load`

`MessageState` can be checkpointed to the session directory and restored on subsequent runs. This is the foundation for long-running, resumable pipelines.

```python
def chatbot(agt: pllm.AgentContext):
    msgs = agt.get_msg_state().load()

    agt.print("Current messages:", msgs)
    out = input("Send a message: ")
    while out:
        msgs.append(out)
        msgs.ask_llm()
        agt.print("Response:", msgs[-1].resolve())
        out = input("Send a message: ")

    msgs.save()
```

## Key methods

| Method | Description |
|---|---|
| `ask_llm(...)` | Adds given documents to the conversation, then gives the entire conversation (MessageState) to the LLM. |
| `ask_functions(...)` | Invoke user functions for any pending function calls in the last response. |
| `ask_human(...)` | Asks the user, then adds their response to the conversation. |
| `save()` | Persist the current message list to disk. |
| `load()` | Load a previously saved message list from disk. |
| `resolve()` | Resolves all LLMResponses in this conversation. |
