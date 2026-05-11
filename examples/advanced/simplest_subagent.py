from dotenv import load_dotenv

import parallem as pllm
import polars as pl


def subagent_app(orch: pllm.AgentOrchestrator):
    collector = []
    # Main agent
    with orch.agent("main-agent") as agt:
        conv = agt.get_msg_state()
        conv.ask_llm(
            "Please name 10 tourist destinations, separated by commas, with no explanation."
        )
        places = [x.strip() for x in conv[-1].final_answer.split(",") if x.strip()]

        # Nested child agent
        for i, item in enumerate(places):
            with orch.agent(f"subagent-{i}") as subagt:
                # Give subagent the parent agent conversation
                subconv = subagt.get_msg_state()
                subconv.extend(conv)  

                subconv.ask_llm(
                    f"Please write 2 paragraphs about {item}.",
                    tools=[pllm.tools.WebSearchTool()],
                )
                collector.append(
                    {
                        "place": item,
                        "description": subconv[-1].final_answer,
                    }
                )
    return pl.DataFrame(collector)


def aggregation_app(orch: pllm.AgentOrchestrator, df: pl.DataFrame):
    all_descriptions = "\n\n".join(df["description"].to_list())
    with orch.agent("agg-agent") as agt:
        conv = agt.get_msg_state()
        conv.ask_llm(
            "I like both urban and countryside exploration. Can you pick the top 3 destinations from the choices below? Please give a brief explanation.",
            all_descriptions,
        )
        return conv[-1].final_answer


if __name__ == "__main__":
    load_dotenv()
    with pllm.resume_directory(".pllm/example/subagents") as orch:
        df = subagent_app(orch)
        with pl.Config(set_ascii_tables=True):
            print(df)

        summary = aggregation_app(orch, df)
        print(summary)
