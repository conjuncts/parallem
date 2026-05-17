import parallem as pllm
from fastmcp import Client
from parallem.core.cast.convert_mcp import mcp_content_block_to_openai_fc_out, mcp_tool_to_openai_responses_tool

mcp_client = Client("examples/mcp/math_server.py")

async def my_agent(agt: pllm.AgentContext, mcp_client: Client):
    conv = agt.get_msg_state()
    
    available_tools = await mcp_client.list_tools()
    schema = [mcp_tool_to_openai_responses_tool(tool) for tool in available_tools]
    last_msg = conv.ask_llm(
        "Add 3 and 5.",
        tools=schema,
    )

    while last_msg.function_calls:
        print(last_msg.function_calls)

        for fc in last_msg.function_calls:
            out = await mcp_client.call_tool(fc.name, fc.args)
            # need to convert ContentBlock
            conv.append(
                pllm.FunctionCallOutput(
                    name=fc.name,
                    call_id=fc.call_id,
                    content=[mcp_content_block_to_openai_fc_out(x) for x in out.content],
                )
            )

        last_msg = conv.ask_llm()
        print(conv[-1].final_answer)

async def main():

    with pllm.resume_directory(
        ".pllm/simplest",
        provider="openai",
        strategy="batch",
        dashboard=True,
        hash_by=["llm"],
        load_dotenv=True,
    ) as orch:
        with orch.agent() as agt:
            async with mcp_client:
                await my_agent(agt, mcp_client)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
