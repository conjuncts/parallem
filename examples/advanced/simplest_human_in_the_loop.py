import logging
from dotenv import load_dotenv

import pipelinellm as pllm

load_dotenv()


def multiply(a: int, b: int) -> int:
    """Calculates a times b."""
    return a * b


def add(a: int, b: int) -> int:
    """Calculates a plus b."""
    return a + b


def divide(a: int, b: int) -> float:
    """Calculates a divided by b."""
    return a / b


with pllm.resume_directory(
    ".pllm/simplest-tool",
    provider="google",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
    hash_by=["llm"],
) as orch:
    with orch.agent() as agt:
        conv = agt.get_msg_state()
        last_msg = conv.ask_llm(
            "Add 3 and 4.",
            tools=pllm.to_tool_schema([multiply, add, divide]),
        )

        while last_msg.resolve_function_calls():
            # Pass all conversation messages to allow caching
            # But do not add it to the conversation history
            permission = agt.ask_human(
                f"Permit function calls: {last_msg.resolve_function_calls()}? (y/n)",
                conv,
            )
            if permission.final_answer.strip().lower() != "y":
                agt.print("Function calls denied by human.")
                continue
            else:
                agt.print("Function calls permitted.")

            conv.ask_functions(multiply=multiply, add=add, divide=divide)
            last_msg = conv.ask_llm()
            agt.print(conv.resolve())

        # ['Add 3 and 4.', '', FunctionCallOutput(name=add, call_id=, content=7...), '3 + 4 = 7']
