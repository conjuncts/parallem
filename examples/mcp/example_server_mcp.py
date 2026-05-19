import parallem as pllm

with pllm.resume_directory(
    ".pllm/simple/mcp",
    strategy="sync",
    provider="openai",
    dashboard=True,
    load_dotenv=True,
) as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm(
            "Roll 2d4+1.",
            tools=[
                pllm.tools.MCPServerTool(
                    server_label="dmcp",
                    server_description="A Dungeons and Dragons MCP server to assist with dice rolling.",
                    server_url="https://dmcp-server.deno.dev/sse",
                    require_approval="never",
                )
            ],
        )

        print(resp.final_answer)
