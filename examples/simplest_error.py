import logging
import os
import pipelinellm as pllm

# load_dotenv()

# Mess up the API key
os.environ["GEMINI_API_KEY"] = "invalid_key"

with pllm.resume_directory(
    ".pllm/simple/error",
    provider="google",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
) as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm("What is 3 cubed?", hash_by=["llm"])

        agt.print(resp.resolve())
