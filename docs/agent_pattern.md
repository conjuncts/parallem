# Agents

## Philosophy

Pipelinellm treats agents a little differently. Traditionally, the agent concept has been tightly coupled with a single LLM: one agent equals one LLM. 

In contrast, pipelinellm associates an agent with a program. What does this mean?

## An agent is just a python function

The most "pipelinic" method way to create a reusable agent is to define a simple **Python function** that takes a `pllm.AgentContext` as a parameter. For example:

```python
--8<-- "examples/simplest_agent.py"
```

But of course, it is just a python function! Nothing is stopping you from embedding all the logic directly in the main function, as is the case in `simplest.py`. We recommend the separate function approach for larger applications, but the choice is really up to you.

## More parameters

Of course, you can always pass more parameters to the agent:
```python
def haiku_agent(agt: pllm.AgentContext, subject: str):
    resp = agt.ask_llm(f"Please write a haiku about {subject}.")
    return resp.final_answer
```

## An agent is more than an LLM

After decoupling the agent concept from one single LLM, it becomes natural to express concepts including:

1. One agent calling multiple LLMs (dynamic model switching, see `simplest_multi.py`):
```python
def polyglot_agent(agt: pllm.AgentContext):
    resp = agt.ask_llm("What is your model name?", llm="gpt-5-nano")
    resp2 = agt.ask_llm("What is your model name?", llm="gemini-2.5-flash")
```

2. Non-linear, branching conversations
```python
def animal_agent(agt: pllm.AgentContext):
    resp = agt.ask_llm("Please simply name 8 animals in a comma-separated list.")
    for animal in resp.split(","):
        resp2 = agt.ask_llm(f"Please write a paragraph on the {animal.final_answer.strip()}.")
```

3. Asking non-LLMs
```python
def best_animal_agent(agt: pllm.AgentContext):
    resp = agt.ask_llm("What is the best animal?")
    resp2 = agt.ask_human(f"Would you agree that {resp.final_answer} is the best animal?")
```

The [Ask API](concepts/ask.md) epitomizes the philosophy that the agent is not an LLM, but a program.

## See also

- [Quickstart](quickstart.md)
- [Ask](concepts/ask.md)
