import logging
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv

from PIL import Image

load_dotenv()

with ParalleLLM.resume_directory(
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

        agt.print(resp.resolve())
