import parallem as pllm
from dotenv import load_dotenv

from PIL import Image

load_dotenv()

with pllm.resume_directory(
    ".pllm/simplest",
    provider="openai",
    strategy="sync",
    dashboard=True,
    hash_by=["llm"],
) as orch:
    for i in range(1):
        with orch.agent(i) as agt:
            img = Image.open(f"experiments/images/animal_{i}.jpg")
            animal = agt.ask_llm("What animal is this?", img)
            agt.print(animal.final_answer)

            haiku = agt.ask_llm(f"Write a haiku about {animal.final_answer}.")
            agt.print(haiku.final_answer)
