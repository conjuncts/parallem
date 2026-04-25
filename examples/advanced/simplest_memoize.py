import logging
import random
import time
import parallem as pllm
from dotenv import load_dotenv


load_dotenv()


def memoized_agent(agt: pllm.AgentContext):
    conv = agt.get_msg_state()
    with agt.memoize() as mem:
        mem.begin()  # Required to start tracking
        agt.print("Long and ardous computation begins...")
        time.sleep(10)
        output = random.randint(1, 100)
        conv.ask_llm(f"In <5 sentences, what is special about the number {output}?")

    agt.print(conv[-1].final_answer)
    # First run: >10 seconds
    # Subsequent runs: instant.
    # Only changes to `conv` is saved.


with pllm.resume_directory(
    ".pllm/simplest",
    provider="google",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
    hash_by=["llm"],
) as orch:
    with orch.agent() as agt:
        memoized_agent(agt)
