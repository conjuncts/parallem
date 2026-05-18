import asyncio

import parallem as pllm


async def haiku_writer_agent(agt: pllm.AgentContext):
    # Declare the agent.
    conv = agt.get_msg_state()
    await conv.ask_llm("Please name an animal in 1 word.")
    await conv.ask_llm(f"Write a haiku about {conv[-1].final_answer}(s).")
    out = conv[-1].final_answer
    print(out)
    return out


async def main():
    with pllm.resume_directory(
        ".pllm/fresh/1/simplest",
        provider="openai",
        strategy="concurrent",
        dashboard=True,
        hash_by=["llm"],
        load_dotenv=True,
    ) as orch:
        # Less good, because Writer-1 must finish before Writer-2 can begin, but still works
        with orch.agent("Writer-1") as a1:
            await haiku_writer_agent(a1)
        with orch.agent("Writer-2") as a2:
            await haiku_writer_agent(a2)

if __name__ == "__main__":
    asyncio.run(main())

    
