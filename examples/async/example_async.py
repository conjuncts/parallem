import asyncio
import random

import parallem as pllm


async def haiku_writer_agent(agt: pllm.AgentContext):
    # Declare the agent.
    conv = agt.get_msg_state()
    
    rand_time = random.uniform(1, 4)
    await asyncio.sleep(rand_time)  # Simulate some processing time
    await conv.ask_llm("Please name an animal in 1 word.")
    await conv.ask_llm(f"Write a haiku about {conv[-1].final_answer}(s).")
    out = conv[-1].final_answer
    print(out)
    return out


async def main():
    with pllm.resume_directory(
        ".pllm/simplest",
        provider="openai",
        strategy="batch",
        dashboard=True,
        load_dotenv=True,
        ask_params={
            "salt": 7,
        }
    ) as orch:
        coros = []
        for i in range(10):
            with orch.agent(f"Writer-{i}") as a:
                coros.append(haiku_writer_agent(a))

        await orch.gather(*coros)

if __name__ == "__main__":
    asyncio.run(main())
