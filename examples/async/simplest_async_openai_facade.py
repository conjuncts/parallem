import asyncio

import parallem as pllm
from dotenv import load_dotenv


async def power_of_n(client, n: int):
    response = await client.responses.create(
        model="gpt-5-nano",
        input=[{"role": "user", "content": f"Name a power of {n}"}],
    )
    return response.output_text


async def run_all(client):
    tasks = [power_of_n(client, n) for n in range(2, 6)]
    return await asyncio.gather(*tasks)


load_dotenv()
with pllm.resume_directory(
    ".pllm/example/async/powers-openai-facade",
    provider="openai",
    strategy="concurrent",
    dashboard=True,
) as orch:
    client = orch.to_client(agent_name="openai-facade")
    out = asyncio.run(run_all(client))
    print(out)
