import logging
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv

load_dotenv()

with ParalleLLM.resume_directory(
    ".pllm/state/conversation",
    provider="openai",
    strategy="sync",
    log_level=logging.DEBUG,
    # ignore_cache=True,
) as pllm:
    with pllm.agent(dashboard=True) as dash:
        msgs = dash.get_msg_state(continuation=True)

        print("Current messages:", msgs)
        out = input("Send a message: ")
        while out:
            msgs.append(out)
            msgs.ask_llm()
            print("Response:", msgs[-1].resolve())
            out = input("Send a message: ")
