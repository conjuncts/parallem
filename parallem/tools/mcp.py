from parallem.types import ServerTool


class MCPTool(ServerTool):
    """Let LLMs use Remote MCP (Model Context Protocol) as a server tool.

    :warning: Experimental."""

    server_tool_type = "mcp"

    def __init__(
        self,
        *,
        server_label: str,
        server_description: str,
        server_url: str,
        require_approval: str,
    ):
        self.server_label = server_label
        self.server_description = server_description
        self.server_url = server_url
        self.require_approval = require_approval
