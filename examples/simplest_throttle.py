import logging
import time
from dotenv import load_dotenv

from parallellm.core.gateway import ParalleLLM
from parallellm.core.throttler import Throttler
from parallellm.types import MinorTweaks

load_dotenv()

with ParalleLLM.resume_directory(
    ".pllm/simple/throttle",
    provider="openai",
    strategy="async",
    log_level=logging.DEBUG,
    ignore_cache=True,
    throttler=Throttler(
        max_requests_per_window=4,
        window_seconds=10,
    ),
    tweaks=MinorTweaks(
        async_max_concurrent=2,
    ),
    dashboard=True,
) as orch:
    time_start = time.time()
    for i in range(5):
        with orch.agent() as agt:
            req_start = time.time()
            resp = agt.ask_llm(f"Please name a power of {i + 2}.", hash_by=["llm"])
            # agt.print(resp.resolve())
            req_end = time.time()
            agt.print(
                f"Response {i} at {req_start - time_start} took {req_end - req_start:.2f}s"
            )
