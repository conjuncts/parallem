# Strategies

`parallem` supports three execution strategies:

- **Sync**: executes each request immediately in normal Python control flow.

- **Batch**: collects requests for provider-side Batch API execution for a 50% discount. ParaLLeM's caching lets you resume where you left off. Great for high-throughput use cases. 

- **Concurrent**: similar to async, but is not true async because `await` and `.final_answer` do not truly yield control to other processes. It still allows requests to be parallelized and executed concurrently.

## See also

- [Quickstart](../quickstart.md)
- [Ask](ask.md)
- [MessageState](msg_state.md)
- [Persistence](persistence.md)
