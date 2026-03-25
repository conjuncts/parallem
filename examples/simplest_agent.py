import parallem as pllm
from dotenv import load_dotenv


load_dotenv()


def haiku_writer_agent(agt: pllm.AgentContext):
    # Example of declaring an agent.
    conv = agt.get_msg_state()
    conv.ask_llm("Please name an animal in 1 word.")
    conv.ask_llm(f"Write a haiku about {conv[-1].final_answer}(s).")
    agt.print(conv)


if __name__ == "__main__":
    with pllm.resume_directory(
        ".pllm/simplest",
        provider="openai",
        strategy="sync",
        dashboard=True,
        hash_by=["llm"],
    ) as orch:
        # Example of instantiating an agent.
        with orch.agent("Yosa Buson") as agt:
            haiku_writer_agent(agt)

        with orch.agent("Matsuo Basho") as agt:
            haiku_writer_agent(agt)
