import logging
import time
from dotenv import load_dotenv

import parallellm as plm

start = time.time()
load_dotenv()

with plm.resume_directory(
    ".pllm/example/tournament-enzy",
    provider="openai",
    strategy="concurrent",
    log_level=logging.DEBUG,
    dashboard=True,
    # ignore_cache=True,
) as orch:
    with orch.agent() as agt:
        agt.print("===Starting Tournament===")
        resp = agt.ask_llm(
            "Please name 8 enzymes. Place your final answer in a code block, separated by newlines.",
        )

        teams = [x for x in resp.resolve().split("```")[1].split("\n")[1:] if x]

        agt.print(f"===Candidates===")
        agt.print(teams)

        while len(teams) > 1:
            agt.print(f"===Round of {len(teams)}===")
            responses = []
            for i in range(0, len(teams), 2):
                if i + 1 < len(teams):
                    resp = agt.ask_llm(
                        f"Given two enzymes, choose the one you like more. Only respond with the name of the enzyme.",
                        teams[i],
                        teams[i + 1],
                    )
                    # do NOT call resp.resolve() in the hot loop
                    responses.append(resp)
                else:
                    # they win by default
                    responses.append(plm.LLMResponse(teams[i]))

            # Resolve only once everything is submitted
            teams = []
            for resp in responses:
                teams.append(resp.resolve())

            agt.print("Teams:", teams)
        agt.print("===Winner===")
        agt.print(teams[0])

print("Total time:", time.time() - start)
# 32 enzymes, sync: 23.34247851371765s
# 32 enzymes: async: 7.341606140136719
# cached: 0.06931805610656738s
