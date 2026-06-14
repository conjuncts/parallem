from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image

import parallem as pllm


class MyModel(BaseModel):
    final_answer: str


def count_files(directory: str) -> int:
    """Counts files in a directory"""
    return 4


def power_of_3_agent(agt: pllm.AgentContext):
    # 1. Basic LLM call
    resp = agt.ask_llm(
        "Please name a power of 3.",
        instructions="No explanation needed.",
    )
    return resp.final_answer.replace("\n", " ")


def web_search_agent(agt: pllm.AgentContext):
    # 2. Web search tool
    resp = agt.ask_llm(
        "In 1 sentence, what is AAPL's current price?",
        tools=[pllm.tools.WebSearchTool()],
    )
    return resp.final_answer.replace("\n", " ")


def structured_output_agent(agt: pllm.AgentContext):
    # 3. Structured output
    resp = agt.ask_llm("What is the capital of France?", structured_output=MyModel)
    return resp.final_answer.replace("\n", " ")


def image_input_agent(agt: pllm.AgentContext):
    # 4. Image input. NOTE: Adjust image as needed.
    img = Image.open("tests/data/images/Nokota_Horses_cropped.jpg")
    img.thumbnail((100, 100))  # Downsample
    resp = agt.ask_llm("What animal is this?", img)
    return resp.final_answer.replace("\n", " ")


def function_calling_agent(agt: pllm.AgentContext):
    # 5,6. Function calling.
    # Keeping track of message state can be tedious. see the MessageState abstraction.
    ls_prompt = "How many files are in ~/examples? Give the final answer in words."
    resp5 = agt.ask_llm(
        ls_prompt,
        tools=[count_files],
    )
    fc_outs = agt.ask_functions(resp5, count_files=count_files)
    resp6 = agt.ask_llm([ls_prompt, resp5, *fc_outs])

    final_answer = ""
    if resp5.function_calls:
        final_answer += f"Function calls: {resp5.function_calls}\n6. "
    final_answer += resp5.final_answer.replace("\n", " ")
    final_answer += resp6.final_answer.replace("\n", " ")
    return final_answer


def file_input_agent(agt: pllm.AgentContext):
    # 7. File input
    file_input = pllm.FileInput(
        filename="example.txt",
        mime_type="text/plain",
        file_content=b"Hello from Amsterdam.",
    )
    resp = agt.ask_llm("What does this file say?", file_input)
    return resp.final_answer.replace("\n", " ")

if __name__ == "__main__":
    load_dotenv()

    with pllm.resume_directory(
        ".pllm/example/batch",
        provider="google",
        strategy="sync",
        dashboard=True,
        llm="gemini-2.5-flash",
        tweaks={"error_mode": "emit"},
        save_input=True,
    ) as orch:
        with orch.agent() as agt:
            print("1. " + power_of_3_agent(agt))
        with orch.agent() as agt:
            print("2. " + web_search_agent(agt))
        with orch.agent() as agt:
            print("3. " + structured_output_agent(agt))
        with orch.agent() as agt:
            print("4. " + image_input_agent(agt))
        with orch.agent() as agt:
            print("5. " + function_calling_agent(agt))
        with orch.agent() as agt:
            print("7. " + file_input_agent(agt))
