# Subagents

In the **subagent pattern**, a main agent creates child subagents to break down and complete a task in parallel. 

This pattern is supported by ParaLLeM:

1. Create a parent agent.
2. Within the parent block, create a child agent. (No special syntax required.)
3. (Optional) Give the child access to its parent's messages.

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

## Dynamically creating subagents

You may want the parent to create a subagent via a function call.

Simply pass in your subagent function as a function call (like any regular function). 

When `ask_functions` is called, ParaLLeM will automatically create the necessary subagents using the names passed into `subagent_names`.

```python
--8<-- "examples/advanced/recursive_subagents.py"
```

## See also

- [Aggregation](aggregation.md)