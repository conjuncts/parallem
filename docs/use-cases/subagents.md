# Subagents

The **subagent pattern** is when a main agent creates many child subagents in order to break down and complete a task in parallel. 

This pattern is supported by pipelinellm: **no special syntax required**. Simply create an agent from within an existing agent context block.

Core pattern:

1. Create a parent conversation.
2. For each item, create a new named agent.
3. Copy parent context into each child with `sub_conv.extend(parent_conv)`.
4. Ask a focused question in each child and collect results.

```python
--8<-- "examples/advanced/simplest_subagent.py"
```

```markdown
[INFO] Resuming with session_id=5
['Paris', 'Tokyo', 'New York', 'Rome', 'Barcelona', 'Istanbul', 'Bangkok', 'Sydney', 'Kyoto', 'Bali']
1. Paris is a city of light and layers, whe...
2. Tokyo is a city where centuries of tradi...
3. New York City, often simply called New Y...
4. Rome, the capital of Italy, is a living ...
5. Barcelona sits on the northeastern coast...
6. Istanbul is a city where two continents ...
7. Bangkok, Thailand's energetic capital, s...
8. Sydney is Australia's largest city, fame...
9. Kyoto, in Japan's Kansai region, long se...
10. Bali, Indonesia's famed island, lies bet...
```

## See also

- [Aggregation](aggregation.md)