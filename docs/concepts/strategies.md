# Strategies

`parallem` supports three execution strategies:

- **Sync**: executes each request immediately: useful for debugging.

- **Batch**: collects requests for provider-side Batch API execution for a 50% discount. Caching lets you resume where you left off. Great for high-throughput use cases. 

- **Async**: executes each request asynchronously with `await` and `async def`.

## See also

- [Quickstart](../quickstart.md)
- [Ask](ask.md)
- [MessageState](msg_state.md)
- [Persistence](persistence.md)
