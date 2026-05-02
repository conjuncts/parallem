We have four principles:

### 1. Batch API first.

The batch API has been historically neglected.
ParaLLeM is designed from the ground up to support the Batch API.

### 2. Synchronous and Batch interchangeable.

By switching from batch to sync, you can debug and prototype without having to wait. 

By switching from sync to batch, you parallelize your agentic workflow (with function calls, structured output, web search, images) to thousands of calls. 

In just one line of code.

### 3. Control flow described by Python code, not data structures.

Python can already express complex conditionals, arbitrary expressions, branching; why use anything else?

### 4. An "agent" is more than an LLM.

We believe that an agent is a *program*. That program happens to use LLMs to automate much of its decision making, but the program can also [ask functions and the user](concepts/ask.md).

By decoupling the LLM from the agent, we allow multi-LLM consensus agents, message history editing, branching conversations, and more.

See the [Your first agent](agent_pattern.md) for more information.


## Developer Experience

We care about developer experience -- from code readability, expressivity, conciseness, to command line pretty-printing and import times. 

We are low-level and lightweight, with friendly abstractions.
