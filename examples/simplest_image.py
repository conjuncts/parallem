import logging
import parallem as pllm
from dotenv import load_dotenv

from PIL import Image

load_dotenv()

with pllm.resume_directory(
    ".pllm/simplest",
    provider="google",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
) as orch:
    with orch.agent() as agt:
        img = Image.open("tests/data/images/Nokota_Horses_cropped.jpg")
        resp = agt.ask_llm("What animal is this?", img)

        print(resp.final_answer)
