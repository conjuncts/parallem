import parallem as pllm
from dotenv import load_dotenv


async def haiku_writer_agent(agt: pllm.AgentContext):
    # Declare the agent.
    conv = agt.get_msg_state()
    conv.ask_llm("Please name an animal in 1 word.")
    conv.ask_llm(f"Write a haiku about {conv[-1].final_answer}(s).")
    out = await conv[-1]
    agt.print(out)
    return out


if __name__ == "__main__":
    load_dotenv()

    with pllm.resume_directory(
        ".pllm/simplest",
        provider="openai",
        strategy="concurrent",
        dashboard=True,
        hash_by=["llm"],
    ) as orch:
        # Instantiate the agent.
        orch.run_agent(haiku_writer_agent, agent_name="Yosa Buson")
        orch.run_agent(haiku_writer_agent, agent_name="Matsuo Basho")
