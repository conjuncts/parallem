import logging

from pydantic import BaseModel
import parallem as pllm


class MyModel(BaseModel):
    final_answer: str


with pllm.resume_directory(
    ".pllm/simplest-tool",
    provider="anthropic",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
    load_dotenv=True,
) as orch:
    with orch.agent() as agt:
        # Structured output
        resp = agt.ask_llm(
            "Please name a power of 3.",
            structured_output=MyModel,
        )

        print(resp.final_json)
