# MCP servers

With ParaLLeM, you can use local MCP servers with the Batch API. Use the `pllm.MCPOutput` object.


```python title="examples/mcp/example_local_mcp.py"
--8<-- "examples/mcp/example_local_mcp.py"
```

```python title="examples/mcp/math_server.py"
--8<-- "examples/mcp/math_server.py"
```

!!! warning
    If MCP outputs are non-deterministic (ie. file system change), then there could be a cache miss, leading to disruption to batched multi-step workflows. Use the `MessageState.save`, `MessageState.load`, and [memoize](../concepts/memoize.md) for non-deterministic blocks.

## Examples

- [examples/mcp/example_local_mcp.py](https://github.com/parallem-ai/parallem/tree/main-prototype/examples/mcp/example_local_mcp.py)
- [examples/mcp/example_server_mcp.py](https://github.com/parallem-ai/parallem/tree/main-prototype/examples/mcp/example_server_mcp.py)
