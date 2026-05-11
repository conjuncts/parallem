# Agents

We treat agents differently. Traditionally, an "agent" has been identified as an "LLM". One agent equals one LLM. 

We believe that an agent is a *program*. This program happens to use LLMs to automate much of its decision making, but the program can also [ask functions and the user](concepts/ask.md).

By decoupling the LLM from the agent, we allow multi-LLM consensus agents, message history editing, branching conversations, and more.

## An agent is just a python function

The most idiomatic way to declare an agent is with a **vanilla Python function** that takes a `pllm.AgentContext`. For example:

```python
--8<-- "examples/simplest_agent.py"
```

It is just a python function! Nothing stops you from embedding all the logic directly in the main function, as is the case in `simplest.py`. The choice is up to you.

## More parameters

You can always pass more parameters to the agent:
```python
def haiku_agent(agt: pllm.AgentContext, subject: str):
    resp = agt.ask_llm(f"Please write a haiku about {subject}.")
    return resp.final_answer
```

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
