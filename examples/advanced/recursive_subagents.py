import parallem as pllm
from dotenv import load_dotenv


def city_summary_agent(agt: pllm.AgentContext, city: str) -> str:
    """Creates an up-to-date summary of a city as a travel destination."""
    conv = agt.get_msg_state()
    conv.ask_llm(f"Write one concise paragraph about {city} as a travel destination.")
    return conv[-1].final_answer


def planner_agent(agt: pllm.AgentContext):
    # Parent agent
    conv = agt.get_msg_state()
    conv.ask_llm(
        "Generate summaries of 3 popular travel destinations.",
        tools=[city_summary_agent],
    )

    # ask_functions creates subagents on the fly
    fc_outs = conv.ask_functions(
        city_summary_agent=city_summary_agent,
        subagent_names=["city-agent-1", "city-agent-2", "city-agent-3"],
    )

    conv.ask_llm(
        "Combine these city summaries into a ranked list.",
        *fc_outs,
    )
    print(conv[-1].final_answer)


if __name__ == "__main__":
    load_dotenv()
    with pllm.resume_directory(
        ".pllm/example/subagent-dynamic",
        dashboard=True,
        provider="google",
    ) as orch:
        with orch.agent("planner") as agt:
            planner_agent(agt)
