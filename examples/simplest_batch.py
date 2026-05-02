import logging
from dotenv import load_dotenv
import parallem as pllm

load_dotenv()

with pllm.resume_directory(
    ".pllm/simplest-batch",
    provider="google",
    strategy="batch",
    log_level=logging.DEBUG,
    dashboard=True,
) as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm("Please name a power of 3.", hash_by=["llm"])

        agt.print(resp.final_answer)
