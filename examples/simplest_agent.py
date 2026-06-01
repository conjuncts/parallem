import parallem as pllm


def syllable_count_agent(agt: pllm.AgentContext, word: str) -> str:
    ct = agt.ask_llm(f'How many syllables are in "{word}"?')
    return ct.final_answer


with pllm.resume_directory(
    ".pllm/example", llm="gpt-5-nano", strategy="sync", load_dotenv=True
) as orch:
    with orch.agent() as agt:
        print(syllable_count_agent(agt, "Hello"))
