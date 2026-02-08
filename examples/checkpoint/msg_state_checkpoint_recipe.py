import logging
import random
from dotenv import load_dotenv
from parallellm.core.gateway import ParalleLLM

load_dotenv()

pllm = ParalleLLM.resume_directory(
    ".pllm/state/recipe",
    provider="openai",
    strategy="sync",
    log_level=logging.DEBUG,
)

# When code is deterministic, LLM calls be can directly cached.

# However, some code will be non-deterministic (ie. API calls, random)
# or might take a really long time, leading to different outcomes.

# In such a case, ParalleLLM introduces "message state"
agent = pllm.agent()

with agent:
    convo = agent.get_msg_state()
    # convo.clear()
    if len(convo) == 0:
        best_vegetable = (
            convo.ask_llm(
                "What is the best vegetable? Enclose your final answer in **double asterisks**."
            )
            .resolve()
            .split("**")[1]
        )
        num_steps = random.randint(3, 5)

        recipe = convo.ask_llm(
            f"Generate a recipe with {num_steps} steps using {best_vegetable}.",
        )
        agent.print(convo)

        # IMPORTANT: need to checkpoint here, because "num-steps" is non-deterministic!
        convo.persist()
    else:
        # Allow user questions, which are not saved along with the conversation
        for item in convo:
            agent.print(item)
        user_input = input("Ask a question about the recipe: ")
        # ie. "What if I don't have an oven?"
        if user_input:
            resp = convo.ask_llm(user_input)
            agent.print("Response:", resp.resolve())

pllm.persist()
