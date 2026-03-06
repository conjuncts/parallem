import parallellm as pllm
from dotenv import load_dotenv

from PIL import Image

load_dotenv()

with pllm.resume_directory(
    ".pllm/simplest",
    provider="google",
    strategy="sync",
    dashboard=True,
) as orch:
    for i in range(1):
        with orch.agent(i) as agt:
            img = Image.open(f"tests/data/images/animal_{i}.jpg")
            animal = agt.ask_llm("What animal is this?", img)
            print(animal.final_answer)

            haiku = agt.ask_llm(f"Write a haiku about {animal.final_answer}.")
            print(haiku.final_answer)
