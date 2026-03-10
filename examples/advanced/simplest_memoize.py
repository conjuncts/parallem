import logging
import random
import time
import pipelinellm as pllm
from dotenv import load_dotenv


load_dotenv()

with pllm.resume_directory(
    ".pllm/simplest",
    provider="google",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
    hash_by=["llm"],
    # ignore_cache=True,
) as orch:
    with orch.agent() as agt:
        conv = agt.get_msg_state()
        with agt.memoize() as mem:
            mem.begin()  # Required to start tracking
            agt.print("Long and ardous computation begins...")
            time.sleep(10)
            output = random.randint(1, 100)
            conv.ask_llm(f"What is special about the number {output}?")

            # This will be memoized
            # That is, any changes to MessageState
            # (and NonMessageState) will be recorded and replayed.

        agt.print(conv[-1].final_answer)
        # If you run this script multiple times, in subsequent runs, the response is instant.
