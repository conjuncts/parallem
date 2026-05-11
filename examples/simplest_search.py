import logging
from dotenv import load_dotenv

import parallem as pllm

load_dotenv()

with pllm.resume_directory(
    ".pllm/simplest",
    provider="google",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
) as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm(
            "In 1 sentence, what is AAPL's current price?",
            tools=[pllm.tools.WebSearchTool()],
            hash_by=["llm"],
        )

        print(resp.final_answer)
