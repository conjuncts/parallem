import logging
import pipelinellm as pllm
from dotenv import load_dotenv

from PIL import Image

load_dotenv()

with pllm.resume_directory(
    ".pllm/simplest",
    provider="google",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
    # ignore_cache=True,
) as orch:
    with orch.agent() as agt:
        img = Image.open("tests/data/images/Nokota_Horses_cropped.jpg")
        resp = agt.ask_llm("What animal is this?", img, hash_by=["llm"])

        agt.print(resp.final_answer)
