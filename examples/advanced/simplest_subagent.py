from dotenv import load_dotenv

import parallem as pllm


def subagent_app(orch: pllm.AgentOrchestrator):
    with orch.agent("main-agent") as agt:
        conv = agt.get_msg_state()
        conv.ask_llm(
            "Please name 10 tourist destinations, separated by commas, with no explanation."
        )
        places = [x.strip() for x in conv[-1].final_answer.split(",") if x.strip()]
        print(places)

        # Subagent pattern: simply create another agent from within the agent
        # No special syntax required.
        descriptions = []
        for i, item in enumerate(places):
            with orch.agent(f"subagent-{i}") as subagt:
                subconv = subagt.get_msg_state()
                subconv.extend(conv)  # Give subagent the parent agent conversation
                subconv.ask_llm(
                    f"Please write 2 paragraphs about {item}.",
                    tools=[pllm.tools.WebSearchTool()],
                )
                descriptions.append(subconv[-1])

        for i, desc in enumerate(descriptions):
            print(f"{i + 1}. {desc.final_answer[:40]}...")


if __name__ == "__main__":
    load_dotenv()
    with pllm.resume_directory(
        ".pllm/example/subagents",
        provider="openai",
        strategy="concurrent",
    ) as orch:
        subagent_app(orch)
