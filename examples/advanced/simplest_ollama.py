from dotenv import load_dotenv
import openai
import parallem as pllm

load_dotenv()
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
)

with pllm.resume_directory(
    ".pllm/simplest",
    provider="openai",
    strategy="sync",
    dashboard=True,
    hash_by=["llm"],
    llm="qwen3.5:4b",
    client=client,
) as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm("Please name a power of 13.", hash_by=["llm"])
        print(resp.final_answer)
