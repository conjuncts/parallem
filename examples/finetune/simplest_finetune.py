import logging
from dotenv import load_dotenv
import parallellm as pllm

load_dotenv()

with pllm.resume_directory(
    ".pllm/simple/finetune",
    provider="openai",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
    # ignore_cache=True,
) as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm(
            "Please name a power of 19.",
            tag="power-of-n",
            save_input=True,
        )

        agt.print(resp.resolve())
