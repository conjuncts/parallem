import asyncio

import parallem as pllm
import openai


async def power_of_n(client: openai.AsyncOpenAI, n: int):
    response = await client.responses.create(
        model="gpt-5-nano",
        input=[{"role": "user", "content": f"Name a power of {n}"}],
    )
    return response.output_text


async def run_all(client: openai.AsyncOpenAI):
    tasks = [power_of_n(client, n) for n in range(2, 6)]
    return await asyncio.gather(*tasks)


with pllm.resume_directory(
    ".pllm/example/openai-client",
    provider="openai",
    strategy="concurrent",
    dashboard=True,
    load_dotenv=True,
) as orch:
    client = orch.to_client(agent_name="facade")
    out = asyncio.run(run_all(client))
    print(out)
