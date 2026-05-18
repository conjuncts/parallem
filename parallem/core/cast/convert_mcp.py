from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.types import Tool

def mcp_tool_to_tool_schema(tool: "Tool") -> dict:
    """Converts MCP tool to OpenAI-compatible JSON schema, for function calling."""
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.inputSchema,
    }
