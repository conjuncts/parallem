import logging
from dotenv import load_dotenv
import parallem as pllm

load_dotenv()

with pllm.resume_directory(
    ".pllm/simplest",
    provider="multi",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
    hash_by=["llm"],
) as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm("What is your model name?")  # gpt-5-nano

        resp2 = agt.ask_llm("What is your model name?", llm="gemini-2.5-flash")
        agt.print([resp.final_answer, resp2.final_answer])
