from dotenv import load_dotenv
import parallem as pllm

load_dotenv()  # Put your OPENAI_API_KEY in the .env file

with pllm.resume_directory(
    ".pllm/simplest",
    provider="openai",
    strategy="sync",
    dashboard=True,
) as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm("Please name a power of 3.", hash_by=["llm"])
        agt.print(resp.final_answer)
