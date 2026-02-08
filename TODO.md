
TODO make seq_id not necessarily auto-increment

explicitly not-agentic philosophy (more of an input/output machine) although agents / responsibility isolation can be implemented with LLMIdentity


- [x] different execution counters
- [x] condition_hash (salt-by)
- [ ] batch api
- [ ] allow LLM to change on a per-`ask_llm` level
    - concoct a "multi-provider" that routes based on `provider_type`
- [ ] dedicated SQLite storage for requests that error
- [ ] retrieve() should also be able to return if a value is pending (in addition to present/absent)
- [ ] "cohort locking": for consistency, if seq_id is "strict" (if we really care that seq_id is consistent across runs), then we need to "lock" based on cohort (wait until all batches in a cohort complete. This can be implemented simply by refusing to proceed - ie. ).
    - this is like a rendezvous in threading
- [x] Automatically persist upon pllm exit
- [ ] accept dict as a LLMDocument


## TODO
- [x] centrally track documents (and incorporate with MessageState) just as responses are also tracked
- [x] tree-based MessageState, which in turn stores all historical MessageState's
    - Solution: existing storage is fine. A Trie can regenerate the MessageState.

- [ ] Error handling: 
    - sync: ask_llm raises an error OR ask_llm produces an error object (ErrorResponse), which is raised when resolve() is called
    - async: ask_llm is fine, but resolve() raises an error
    - mode 1: exceptions are fatal
    - mode 2: log exceptions and continue
    - three error handling modes: None, skip, retry (exponential backoff)
    - Better errors when a bad request (ie. openai.BadRequest) is made



Input storage:
- [x] doc_hash <=> list of message hashes (doc_table)
- [x] message_hash <=> message_value (message_table)
- [x] tool calls for batch mode
- [x] image as valid document type
- [ ] fix tag for batches
- [ ] roll up messages when several consecutive come from the same role

- [ ] the doc_hash/msg_hash naming convention is kinda backward due to history

- [ ] resolve_all
- [ ] export_all
- [ ] import batch.zip

- [ ] make arguments that work on all ask_llm calls (ie. save_input)
    - save_input
    - hash_by
- [ ] a ParquetWriter that is backed by a temp SQLite table
- [ ] in MessageState, is there some more elegant way to only keep track of the deltas to the conversation?
    - YES. Implies keeping track of msg_hashes alongside true messages. When Then, pass "False" to the agent's ask_llm method. Use a special class to keep track of this shenanigan which only contains 
        - Ensure that append(), pop() all appropriately track both the msg and the metadata (msg_hash)
        - Problem: LLMResponse needs to be resolved to be given a hash. :/
        - Solution: all LLMResponses can deterministically be given the hash "{agent_name}:{seq_id}:{sess_id}". But then we will need to ensure that LLMResponses have their "sess_id" always restored.
            This also cleans up the mess between strict_documents and documents in agent.py
- [ ] Ensure that "sess_id" is always restored.
- [ ] Read up on MCP. (Maybe an ask_mcp function for easy integration)
    - In the docs, emphasize more the fact that plug-and-play is allowed
    - why is it called "ask_functions"? Because, the LLM asks functions some stuff, and it gives a response.
    - why is it called "ask_mcp"? Because, the LLM asks the MCP server to do some stuff, and it gives a response


Read
- MCP
- Claude Skills
    - https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf?hsLang=en
- Langchain's persistence layer:
    - https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf?hsLang=en
- OpenRouter


- human-in-the-loop: `ask_human` :)
    - emits a HumanResponse

- what if a function call also involves a LLM? well then the function will need to take in an agent object. Then you will need to do `functools.partial(my_func, agent)`. TODO: Consider then doing some hacking where ask_functions() automatically injects the *first* argument of type AgentContext (dependency inejction) (syntactic sugar)

- [ ] fix that ReadyLLMResponse don't have the original sess_id. pertinent: ParsedResponse should be modified to contain (seq_id, sess_id).
- [ ] need to hash based on available tools??? (TODO: issue a warning)
- [ ] material docs
- [ ] restore resolve_json()
- [ ] continuable errors (ie. JSON)
- [ ] dashboard should be placed at the 'pllm' level, not the agent level, to avoid spam


```python
with pllm.conversation(name=...) as conv:
```