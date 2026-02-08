import logging
from dotenv import load_dotenv

from parallellm.core.gateway import ParalleLLM
from parallellm.tools.server import WebSearchTool

load_dotenv()

with ParalleLLM.resume_directory(
    ".pllm/simplest",
    provider="google",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
    # ignore_cache=True,
) as pllm:
    # with pllm.default():
    with pllm.agent() as agt:
        resp = agt.ask_llm(
            "In 1 sentence, what is AAPL's current price?",
            # llm="claude-haiku-4-5-20251001",
            tools=[WebSearchTool()],
            hash_by=["llm"],
        )

        agt.print(resp.resolve())


# pllm.persist()
