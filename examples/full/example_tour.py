import logging
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image

import pipelinellm as pllm

load_dotenv()


class MyModel(BaseModel):
    final_answer: str


def count_files(directory: str) -> int:
    """Counts files in a directory"""
    return 4


with pllm.resume_directory(
    ".pllm/example/batch",
    provider="anthropic",
    strategy="batch",
    log_level=logging.DEBUG,
    dashboard=True,
    hash_by=["llm"],
    llm="claude-haiku-4-5-20251001",
) as orch:
    with orch.agent() as agt:
        # 1. Basic LLM call
        resp1 = agt.ask_llm("Please name a power of 3.")

        # 2. Web search tool
        resp2 = agt.ask_llm(
            "In 1 sentence, what is AAPL's current price?",
            tools=[pllm.tools.WebSearchTool()],
        )

        # 3. Structured output
        resp3 = agt.ask_llm("What is the capital of France?", text_format=MyModel)

        # 4. Image input. NOTE: Adjust image as needed.
        img = Image.open("tests/data/images/Nokota_Horses_cropped.jpg")
        img.thumbnail((100, 100))  # Downsample
        resp4 = agt.ask_llm("What animal is this?", img)

        # 5,6. Function calling.
        # Keeping track of message state can be tedious. see the MessageState abstraction.
        ls_prompt = "How many files are in ~/examples? Give the final answer in words."
        resp5 = agt.ask_llm(
            ls_prompt,
            tools=pllm.to_tool_schema([count_files]),
        )
        fc_outs = agt.ask_functions(resp5, count_files=count_files)
        resp6 = agt.ask_llm([ls_prompt, resp5, *fc_outs])

        for i, resp in enumerate([resp1, resp2, resp3, resp4, resp5, resp6]):
            if fcs := resp.resolve_function_calls():
                agt.print(fcs)
            final_answer = resp.final_answer.replace("\n", " ")
            agt.print(f"{i + 1}. {final_answer}")
