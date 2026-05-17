from typing import Optional

from parallem.types import ServerTool


class MCPServerTool(ServerTool):
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
        authorization_token: Optional[str] = None,
        default_config: Optional[dict] = None,
        configs: Optional[dict] = None,
        cache_control: Optional[dict] = None,
    ):
        self.server_label = server_label
        self.server_description = server_description
        self.server_url = server_url
        self.require_approval = require_approval
        self.authorization_token = authorization_token
        self.default_config = default_config
        self.configs = configs
        self.cache_control = cache_control
        self.kwargs = {
            "authorization_token": authorization_token,
            "default_config": default_config,
            "configs": configs,
            "cache_control": cache_control,
        }
