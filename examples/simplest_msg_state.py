import parallem as pllm
from dotenv import load_dotenv


load_dotenv()

with pllm.resume_directory(
    ".pllm/simplest",
    provider="openai",
    strategy="sync",
    dashboard=True,
    hash_by=["llm"],
) as orch:
    for i in range(1):
        with orch.agent(i) as agt:
            conv = agt.get_msg_state()
            conv.ask_llm("Please name an animal in 1 word.")
            conv.ask_llm(f"Write a haiku about {conv[-1].final_answer}(s).")
            print(conv)

# ['Please name an animal in 1 word.', ReadyLLMResponse('lion', doc_hash=69237628), 'Write a haiku about lion(s).', ReadyLLMResponse('Golden savanna\nLions rest beneath the stars\nRoa...', doc_hash=947e27c0)]
