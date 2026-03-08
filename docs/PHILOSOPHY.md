# Principles

pipelinellm is based on four principles:

## 1. Batch API first.

Historically, the batch API has been neglected even by major production LLM libraries. Meanwhile, pipelinellm is designed from the ground up to support the Batch API.

## 2. Synchronous and Batch interchangeable.

By instantly switching from batch to sync, you can debug batch pipelines and allow rapid prototyping without having to wait. 

Switching from sync to batch, you effortlessly scale up your complex workflows (ie. function calls, structured output, web search, images) to hundreds of thousands of calls: they instantly translate to the Batch API.

## 3. Workflows described by Python code, not data structures.

Python can already express complex conditionals, arbitrary expressions, branching; why use anything else?

## 4. An "agent" should be a *program*, not an LLM. 

It just so happens that the agent uses LLM(s) to automate much of its decision making. But the agent can also ask functions and the user. This philosophy is represented in the `ask` (`ask_llm`, `ask_functions`, `ask_human`) API. 

By decoupling the LLM from the agent, we conceptually allow multi-LLM pipelines, message history editing, non-linear message histories, and more.


# Developer Experience

Finally, we deeply care about developer experience. This impacts everything from code readability, expressivity, conciseness, to command line pretty-printing and import times. 

We are low-level and lightweight. We hope that our abstractions are intuitive and non-obtrusive.
