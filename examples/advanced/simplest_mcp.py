from dotenv import load_dotenv
import parallellm as plm

load_dotenv()

with plm.resume_directory(
    ".pllm/simple/mcp",
    strategy="sync",
    provider="openai",
    # ignore_cache=True,
    # rewrite_cache=True,
    dashboard=True,
) as orch:
    with orch.agent() as agt:
        # https://platform.openai.com/docs/guides/tools-connectors-mcp
        # openai - SSE is ok
        # google - HTTP only (not SSE)
        resp = agt.ask_llm(
            "Roll 2d4+1.",
            tools=[
                plm.tools.MCPTool(
                    server_label="dmcp",
                    server_description="A Dungeons and Dragons MCP server to assist with dice rolling.",
                    server_url="https://dmcp-server.deno.dev/sse",
                    require_approval="never",
                )
            ],
        )

        agt.print(resp.resolve())
