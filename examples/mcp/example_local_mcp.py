import parallem as pllm
from fastmcp import Client
from parallem.core.cast.convert_mcp import mcp_tool_to_tool_schema

mcp_client = Client("examples/mcp/math_server.py")


async def my_agent(agt: pllm.AgentContext, mcp_client: Client, question: str):
    conv = agt.get_msg_state()

    available_tools = await mcp_client.list_tools()
    schema = [mcp_tool_to_tool_schema(tool) for tool in available_tools]
    last_msg = conv.ask_llm(
        question,
        tools=schema,
    )

    while last_msg.function_calls:
        print(last_msg.function_calls)

        for fc in last_msg.function_calls:
            out = await mcp_client.call_tool(fc.name, fc.args)
            conv.append(
                pllm.MCPOutput(
                    name=fc.name,
                    call_id=fc.call_id,
                    content=out.content,
                )
            )

        last_msg = conv.ask_llm()
        print(conv[-1].final_answer)


async def main():
    with pllm.resume_directory(
        ".pllm/simplest",
        provider="anthropic",
        strategy="sync",
        dashboard=True,
        load_dotenv=True,
    ) as orch:
        with orch.agent() as agt:
            async with mcp_client:
                await my_agent(agt, mcp_client, "Add 3 and 6.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
