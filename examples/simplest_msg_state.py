import logging
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv

load_dotenv()

with ParalleLLM.resume_directory(
    ".pllm/state/msg-state",
    provider="openai",
    strategy="sync",
    log_level=logging.DEBUG,
    # ignore_cache=True,
) as pllm:
    with pllm.agent(dashboard=True) as dash:
        msgs = dash.get_msg_state(continuation=True)

        print("Current messages:", msgs)
        out = input("Send a message: ")
        if out:
            msgs.append(out)
            resp = dash.ask_llm(msgs)
            print("Response:", resp.resolve())
            msgs.append(resp)
