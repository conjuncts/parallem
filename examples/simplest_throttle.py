import logging
import time
from dotenv import load_dotenv

import parallem as pllm

load_dotenv()

with pllm.resume_directory(
    ".pllm/simple/throttle",
    provider="openai",
    strategy="concurrent",
    log_level=logging.DEBUG,
    ignore_cache=True,
    throttler=pllm.Throttler(
        max_requests_per_window=4,
        window_seconds=10,
    ),
    tweaks={
        "max_concurrent": 2,
    },
    dashboard=True,
) as orch:
    time_start = time.time()
    for i in range(5):
        with orch.agent() as agt:
            req_start = time.time()
            resp = agt.ask_llm(f"Please name a power of {i + 2}.", hash_by=["llm"])
            # agt.print(resp.final_answer)
            req_end = time.time()
            agt.print(
                f"Response {i} at {req_start - time_start} took {req_end - req_start:.2f}s"
            )
