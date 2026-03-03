import logging

from dotenv import load_dotenv
import parallellm as plm

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


with plm.resume_directory(
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
        msgs = ["How many files are in '~/examples'? Give the final answer in words."]
        resp = agt.ask_llm(
            msgs,
            tools=tools,
        )

        agt.print(resp.resolve())
        tool_calls = resp.resolve_function_calls()
        for call in tool_calls:
            agt.print(
                f"Tool call: `{call.name}` with args {call.args} call_id {call.call_id}"
            )

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "count_files"
        msgs.append(resp)

        computed_tool_output = plm.FunctionCallOutput(
            name=tool_calls[0].name,
            content=ls_tool(tool_calls[0].args),
            call_id=tool_calls[0].call_id,
        )

        resp = agt.ask_llm(msgs + [computed_tool_output])
        agt.print(resp.resolve())
