# Agents

Conventional wisdom holds that *"one agent = one LLM"*. But we believe that an agent is a *program*. This program happens to use LLMs to automate much of its decision making, but the program can also [ask functions](concepts/ask.md) and the [user](concepts/ask.md).

As a result, agents in ParaLLeM are more powerful.

## An agent is just a python function

The most idiomatic way to declare an agent is with a **vanilla Python function** that takes a `pllm.AgentContext`. For example:

```python
--8<-- "examples/simplest_agent.py"
```

It is just a python function, nothing special!

## An agent is more than an LLM

After decoupling the agent concept from one single LLM, it becomes natural to express concepts including:

1. One agent calling multiple LLMs (dynamic model switching):
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

## See also

- [Quickstart](quickstart.md)
- [Ask](concepts/ask.md)
