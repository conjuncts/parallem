import parallem as pllm
from dotenv import load_dotenv


async def haiku_writer_agent(agt: pllm.AgentContext):
    # Declare the agent.
    conv = agt.get_msg_state()
    await conv.ask_llm("Please name an animal in 1 word.")
    await conv.ask_llm(f"Write a haiku about {conv[-1].final_answer}(s).")
    out = conv[-1].final_answer
    print(out)
    return out


if __name__ == "__main__":
    load_dotenv()

    with pllm.resume_directory(
        ".pllm/simplest",
        provider="openai",
        strategy="async",
        dashboard=True,
    ) as orch:
        # Instantiate the agent.
        a1 = orch.create_agent(haiku_writer_agent, agent_name="Writer-1")
        a2 = orch.create_agent(haiku_writer_agent, agent_name="Writer-2")

        # Run agents. Similar to async.gather or async.TaskGroup
        out = orch.run_agents(a1, a2)
