Underlying parallellm are 4 design philosophies:

1. **Batch first.** parallellm is great for when a response is not immediately needed; or OLAP workflows. 

2. **Sync and batch interchangeable**. By instantly switching from batch to sync, you can debug batch pipelines and allow rapid prototyping without having to wait. Switching from sync to batch, you effortlessly scale up to hundreds of thousands of calls.

3. **Control flow is best expressed with code, not data structures.** Python can already express complex conditionals, arbitrary expressions, branching; why use anything else?

4. **An "agent" should be a *program*, not a LLM.** It just so happens that the agent uses LLM(s) to automate much of its decision making.But the agent can also ask functions and the user. This philosophy is represented in the `ask` (`ask_llm`, `ask_functions`, `ask_human`) API.
