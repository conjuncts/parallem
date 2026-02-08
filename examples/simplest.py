import logging
from dotenv import load_dotenv
from parallellm.core.gateway import ParalleLLM

load_dotenv()

with ParalleLLM.resume_directory(
    ".pllm/simplest",
    provider="openai",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
    # ignore_cache=True,
) as pllm:
    with pllm.agent() as agt:
        resp = agt.ask_llm("Please name a power of 3.", hash_by=["llm"])

        agt.print(resp.resolve())


# pllm.persist()
