
TODO make seq_id not necessarily auto-increment

explicitly not-agentic philosophy (more of an input/output machine) although agents / responsibility isolation can be implemented with LLMIdentity


- [ ] dedicated SQLite storage for requests that error
- [ ] retrieve() should also be able to return if a value is pending (in addition to present/absent)
- [ ] "cohort locking": for consistency, if seq_id is "strict" (if we really care that seq_id is consistent across runs), then we need to "lock" based on cohort (wait until all batches in a cohort complete. This can be implemented simply by refusing to proceed - ie. ).
    - this is like a rendezvous in threading
- [ ] accept dict as a LLMDocument

## TODO

- [x] Regenerate tree-based history of MessageState with a Trie.
- [ ] Error handling: 
    - sync: ask_llm raises an error OR ask_llm produces an error object (ErrorResponse), which is raised when resolve() is called
    - Concurrent: ask_llm is fine, but resolve() raises an error
    - mode 1: exceptions are fatal
    - mode 2: log exceptions and continue
    - three error handling modes: None, skip, retry (exponential backoff)
    - Better errors when a bad request (ie. openai.BadRequest) is made



Input storage:
- [ ] for store_input, switch to SQLite: probably more performant.
- [ ] roll up messages when several consecutive come from the same role


- [ ] export_all

- [x] make arguments that work on all ask_llm calls (ie. save_input)
    - save_input
    - hash_by
    - llm
    - already exists: ask_params
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

- what if a function call also involves a LLM? well then the function will need to take in an agent object. Then you will need to do `functools.partial(my_func, agent)`. TODO: Consider then doing some hacking where ask_functions() automatically injects the *first* argument of type AgentContext (dependency inejction) (syntactic sugar)

- [ ] need to hash based on available tools??? (TODO: issue a warning)
- [ ] material docs
- [ ] continuable errors (ie. JSON)


- output_text, final_answer

- Manual MCP server
    - ping get tool calls 
- dry run

High priority:
- [ ] batch bulk import/export (batch.zip)
    - not sure it's possible
- [x] df export to parquet
- [ ] clienet as a way to delegate
- [x] tags for batch mode
- [ ] demonstrate that traditional workflows (ie. a simple ChatGPT wrapper) is also possible with pipelinellm (albeit not the intended purpose). Probably needs to use some message uuid as the agent_name.
- [ ] demonstrate that subagents (nested agents) is possible.
- [x] instead of locking up, do not resend responses that are known to already be pending. So create a PendingBatchResponse() interrupt.
- [ ] In memory datastore.
- The "one-user" problem. "true" async / multiprocessing support.

- If (sess_id, seq_id) serves as a unique key, then response_id can be removed from the main table.
- If upserting, preserve old (sess_id, seq_id).
