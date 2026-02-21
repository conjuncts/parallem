import logging

from pydantic import BaseModel
from dotenv import load_dotenv
import parallellm as plm

load_dotenv()


class MyModel(BaseModel):
    final_answer: str


with plm.resume_directory(
    ".pllm/simplest-tool",
    provider="google",
    strategy="batch",
    log_level=logging.DEBUG,
    dashboard=True,
    # ignore_cache=True,
) as orch:
    with orch.agent() as agt:
        # Structured output
        resp = agt.ask_llm(
            "Please name a power of 3.", hash_by=["llm"], text_format=MyModel
        )

        agt.print(resp.resolve())
