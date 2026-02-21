import logging
from dotenv import load_dotenv
import parallellm as plm

load_dotenv()

orch = plm.resume_directory(
    ".pllm/example/nfl",
    provider="openai",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
)

with orch.agent() as agt:
    agt.print("This will always be executed")

    resp = agt.ask_llm(
        "Please name 8 NFL teams. Place your final answer in a code block, separated by newlines.",
    )

    teams = resp.resolve().split("```")[1].split("\n")[1:9]

    agt.print(f"Got teams: {teams}")

    games = []
    for i in range(0, len(teams), 2):
        resp = agt.ask_llm(
            f"Given a game between the {teams[i]} and the {teams[i + 1]}, simply predict the winner and the score.",
        )
        agt.print("Asked!")
        # do NOT call resp.resolve() in the hot loop
        games.append(resp)

    # waiting for all to be submitted results in better batching!
    game_descriptions = []
    for resp in games:
        game_descriptions.append(resp.resolve())

    agt.print("Descriptions:", [x[:70] for x in game_descriptions])


plm.persist()
