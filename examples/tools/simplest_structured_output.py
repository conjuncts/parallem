import logging

from pydantic import BaseModel
from dotenv import load_dotenv
import parallem as pllm

load_dotenv()


class MyModel(BaseModel):
    final_answer: str


with pllm.resume_directory(
    ".pllm/simplest-tool",
    provider="anthropic",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
    hash_by=["llm"],
) as orch:
    with orch.agent() as agt:
        # Structured output
        resp = agt.ask_llm(
            "Please name a power of 3.",
            structured_output=MyModel,
        )

        agt.print(resp.resolve_json())
