import parallem as pllm


def add(a: int, b: int) -> int:
    """Calculates a plus b."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Calculates a minus b."""
    return a - b


with pllm.resume_directory(
    ".pllm/simplest-tool",
    provider="google",
    strategy="sync",
    dashboard=True,
    load_dotenv=True,
) as orch:
    with orch.agent() as agt:
        conv = agt.get_msg_state()
        last_msg = conv.ask_llm(
            "Add 3 and 4.",
            tools=[add, subtract],
            # Or explicitly pass OpenAPI-compliant schemas as list of dicts
            # tools=pllm.to_tool_schema([add, subtract]),
        )

        while last_msg.function_calls:
            print(last_msg.function_calls)
            conv.ask_functions(add=add, subtract=subtract)
            last_msg = conv.ask_llm()
            print(conv[-1].final_answer)
