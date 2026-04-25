import logging
from dotenv import load_dotenv
import parallem as pllm

load_dotenv()

with pllm.resume_directory(
    ".pllm/example/chatbot",
    provider="openai",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
) as orch:
    with orch.agent() as agt:
        msgs = agt.get_msg_state().load()

        agt.print("Current messages:", msgs)
        out = input("Send a message: ")
        while out:
            msgs.append(out)
            msgs.ask_llm()
            agt.print("Response:", msgs[-1].resolve())
            out = input("Send a message: ")

        msgs.save()
