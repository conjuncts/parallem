import logging
from dotenv import load_dotenv

import parallellm as plm

load_dotenv()


def multiply(a: int, b: int) -> int:
    """Calculates a times b."""
    return str(a * b)


def add(a: int, b: int) -> int:
    """Calculates a plus b."""
    return str(a + b)


def divide(a: int, b: int) -> float:
    """Calculates a divided by b."""
    return str(a / b)


with plm.resume_directory(
    ".pllm/simplest-tool",
    provider="google",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
    # ignore_cache=True,
) as orch:
    with orch.agent() as agt:
        # See docs on the MessageState abstraction.
        convo = agt.get_msg_state()
        resp = convo.ask_llm(
            "Add 3 and 4.",
            hash_by=["llm"],
            tools=plm.to_tool_schema([multiply, add, divide]),
        )

        convo.ask_functions(multiply=multiply, add=add, divide=divide)
        convo.ask_llm(hash_by=["llm"])
        agt.print(convo.resolve())

        # ['Add 3 and 4.', '', FunctionCallOutput(name=add, call_id=, content=7...), '3 + 4 = 7']
