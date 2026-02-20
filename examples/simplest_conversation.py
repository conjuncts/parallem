import logging
from dotenv import load_dotenv
from parallellm.core.gateway import ParalleLLM

load_dotenv()

with ParalleLLM.resume_directory(
    ".pllm/state/conversation",
    provider="openai",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
    # ignore_cache=True,
) as orch:
    with orch.agent() as agt:
        msgs = agt.get_msg_state(continuation=True)

        agt.print("Current messages:", msgs)
        out = input("Send a message: ")
        while out:
            msgs.append(out)
            msgs.ask_llm()
            agt.print("Response:", msgs[-1].resolve())
            out = input("Send a message: ")
