from pipelinellm.types import ServerTool


class WebSearchTool(ServerTool):
    """Let LLMs have web search functionality."""

    server_tool_type = "web_search"

    def __init__(self, kwargs: dict = None):
        self.kwargs = kwargs or {}

    # TODO: additional parameters website whitelist filter (openai)


class CodeInterpreterTool(ServerTool):
    """Let LLMs interpret and execute code."""

    server_tool_type = "code_interpreter"

    def __init__(self, kwargs: dict = None):
        self.kwargs = kwargs or {"type": "auto", "memory_limit": "1g"}
