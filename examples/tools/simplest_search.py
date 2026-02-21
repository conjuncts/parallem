import logging
from dotenv import load_dotenv

import parallellm as plm

load_dotenv()

with plm.resume_directory(
    ".pllm/simplest",
    provider="google",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
    # ignore_cache=True,
) as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm(
            "In 1 sentence, what is AAPL's current price?",
            # llm="claude-haiku-4-5-20251001",
            tools=[plm.tools.WebSearchTool()],
            hash_by=["llm"],
        )

        agt.print(resp.resolve())


# pllm.persist()
