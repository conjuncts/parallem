import logging

from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image

import parallellm as pllm

load_dotenv()


class MyModel(BaseModel):
    final_answer: str


tools = [
    {
        "type": "function",
        "name": "count_files",
        "description": "Count the number of files in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "The path to the directory to count files in.",
                },
            },
            "required": ["directory"],
        },
    }
]

with pllm.resume_directory(
    ".pllm/example/batch",
    provider="google",
    strategy="batch",
    log_level=logging.DEBUG,
    dashboard=True,
    # ignore_cache=True,
) as orch:
    with orch.agent() as agt:
        resp1 = agt.ask_llm("Please name a power of 3.", hash_by=["llm"])
        resp2 = agt.ask_llm(
            "In 1 sentence, what is AAPL's current price?",
            # llm="claude-haiku-4-5-20251001",
            tools=[pllm.tools.WebSearchTool()],
            hash_by=["llm"],
        )
        resp3 = agt.ask_llm(
            "How many files are in ~/examples? Give the final answer in words.",
            hash_by=["llm"],
            tools=tools,
        )
        resp4 = agt.ask_llm(
            "What is the capital of France?", hash_by=["llm"], text_format=MyModel
        )

        img = Image.open("tests/data/images/Nokota_Horses_cropped.jpg")
        resp5 = agt.ask_llm("What animal is this?", img, hash_by=["llm"])

        for resp in [resp1, resp2, resp3, resp4, resp5]:
            if fcs := resp.resolve_function_calls():
                for fc in fcs:
                    agt.print(f"function_call {fc.name} {fc.args} {fc.call_id}")
            agt.print(resp.resolve())
