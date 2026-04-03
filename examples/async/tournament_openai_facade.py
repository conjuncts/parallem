import asyncio
import logging

from dotenv import load_dotenv
from parallem.core.gateway import resume_directory

load_dotenv()


def parse_codeblock_lines(text: str) -> list[str]:
    chunks = text.split("```")
    if len(chunks) < 2:
        return [line.strip() for line in text.splitlines() if line.strip()]
    block = chunks[1]
    return [line.strip() for line in block.splitlines()[1:] if line.strip()]


async def nfl_tournament() -> None:
    with resume_directory(
        ".pllm/example/nfl-openai-facade",
        provider="openai",
        strategy="concurrent",
        log_level=logging.DEBUG,
        dashboard=True,
    ) as orch:
        client = orch.to_client(agent_name="nfl-openai-facade")

        resp = await client.responses.create(
            model="gpt-5-nano",
            input=[
                {
                    "role": "user",
                    "content": "Please name 8 NFL teams. Place your final answer in a code block, separated by newlines.",
                }
            ],
        )

        teams = parse_codeblock_lines(resp.output_text)
        print("===Candidates===")
        print(teams)

        while len(teams) > 1:
            print(f"===Round of {len(teams)}===")
            tasks = []
            winners = []

            for i in range(0, len(teams), 2):
                if i + 1 < len(teams):
                    tasks.append(
                        client.responses.create(
                            model="gpt-5-nano",
                            input=[
                                {
                                    "role": "user",
                                    "content": f"Given a game between the {teams[i]} and the {teams[i + 1]}, simply predict the winner.",
                                }
                            ],
                        )
                    )
                else:
                    winners.append(teams[i])

            responses = await asyncio.gather(*tasks)
            winners.extend(resp.output_text.strip() for resp in responses)

            teams = winners
            print("Teams:", teams)


if __name__ == "__main__":
    asyncio.run(nfl_tournament())
