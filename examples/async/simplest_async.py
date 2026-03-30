import parallem as pllm
from dotenv import load_dotenv


async def power_of_n_agent(agt: pllm.AgentContext, n: int):
    return await agt.ask_llm(f"Name a power of {n}")


load_dotenv()
with pllm.resume_directory(
    ".pllm/example/async/powers",
    strategy="batch",  # or concurrent, batch
    dashboard=True,
) as orch:
    agts = []
    for i in range(2, 6):
        # Instantiate the agent.
        agts.append(orch.create_agent(power_of_n_agent, i))

    # Run agents. Similar to async.gather
    out = orch.run_agents(agts)
    print(out)
