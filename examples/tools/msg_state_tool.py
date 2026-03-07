"""
In comparison to `simplest_tool.py`, this example demonstrates how MessageState
to conveniently store conversation history.
"""

import logging

from dotenv import load_dotenv
import pipelinellm as pllm

load_dotenv()


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


def ls_tool(directory) -> str:
    return f"There are 4 files in {directory}."


with pllm.resume_directory(
    ".pllm/simplest-tool",
    provider="openai",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
    hash_by=["llm"],
    # ignore_cache=True,
) as orch:
    with orch.agent() as agt:
        # Tools
        convo = agt.get_msg_state()
        resp = convo.ask_llm(
            "How many files are in '~/examples'? Give the final answer in words.",
            tools=tools,
            # llm="gpt-4o"
        )

        tool_calls = resp.resolve_function_calls()
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "count_files"
        convo.ask_functions(count_files=ls_tool)
        convo.ask_llm()
        agt.print(convo.resolve())
