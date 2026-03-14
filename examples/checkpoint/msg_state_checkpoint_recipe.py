import logging
import random
from dotenv import load_dotenv
import pipelinellm as pllm

load_dotenv()


def recipe_example(agent: pllm.AgentContext):
    conv = agent.get_msg_state()
    # conv.clear()
    if len(conv) == 0:
        best_vegetable = conv.ask_llm(
            "What is the best vegetable? Enclose your final answer in **double asterisks**."
        ).final_answer.split("**")[1]
        num_steps = random.randint(3, 5)
        conv.ask_llm(
            f"Generate a recipe with {num_steps} steps using {best_vegetable}.",
        )
        agent.print(conv)
        conv.save()
    else:
        # Allow user questions, which are not saved along with the conversation
        for item in conv:
            agent.print(item)
        user_input = input("Ask a question about the recipe: ")
        # ie. "What if I don't have an oven?"
        if user_input:
            resp = conv.ask_llm(user_input)
            agent.print("Response:", resp.final_answer)


if __name__ == "__main__":
    with pllm.resume_directory(
        ".pllm/recipe",
        provider="openai",
        strategy="sync",
        log_level=logging.DEBUG,
        dashboard=True,
        llm="gpt-5-mini-2025-08-07",
        hash_by=["llm"],
        ignore_cache=True,
    ) as orch:
        with orch.agent() as agt:
            recipe_example(agt)
