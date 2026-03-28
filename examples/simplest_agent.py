import parallem as pllm
from dotenv import load_dotenv


def haiku_writer_agent(agt: pllm.AgentContext):
    # Declare the agent.
    conv = agt.get_msg_state()
    conv.ask_llm("Please name an animal in 1 word.")
    conv.ask_llm(f"Write a haiku about {conv[-1].final_answer}(s).")
    return conv[-1].final_answer


if __name__ == "__main__":
    load_dotenv()

    with pllm.resume_directory(
        ".pllm/simplest",
        provider="openai",
        strategy="sync",
        dashboard=True,
        hash_by=["llm"],
    ) as orch:
        # Instantiate the agent.
        with orch.agent("Yosa Buson") as agt:
            agt.print(haiku_writer_agent(agt))

        with orch.agent("Matsuo Basho") as agt:
            agt.print(haiku_writer_agent(agt))
