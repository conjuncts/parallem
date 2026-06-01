import logging
from dotenv import load_dotenv
import parallem as pllm

load_dotenv()


def nfl_tournament(agt: pllm.AgentContext):
    resp = agt.ask_llm(
        "Please name 8 NFL teams. Place your final answer in a code block, separated by newlines.",
    )

    teams = resp.final_answer.split("```")[1].split("\n")[1:9]

    print(f"Got teams: {teams}")
    teams = [x for x in resp.final_answer.split("```")[1].split("\n")[1:] if x]

    print("===Candidates===")
    print(teams)

    while len(teams) > 1:
        print(f"===Round of {len(teams)}===")
        responses = []
        for i in range(0, len(teams), 2):
            if i + 1 < len(teams):
                resp = agt.ask_llm(
                    f"Given a game between the {teams[i]} and the {teams[i + 1]}, simply predict the winner.",
                    teams[i],
                    teams[i + 1],
                )
                # do NOT call resp.final_answer in the hot loop
                responses.append(resp)
            else:
                # they win by default
                responses.append(pllm.LLMResponse(teams[i]))

        # Resolve only once everything is submitted
        teams = []
        for resp in responses:
            teams.append(resp.final_answer)
        print("Teams:", teams)


with pllm.resume_directory(
    ".pllm/example/nfl",
    provider="openai",
    strategy="async",
    log_level=logging.DEBUG,
    dashboard=True,
) as orch:
    with orch.agent() as agt:
        nfl_tournament(agt)
