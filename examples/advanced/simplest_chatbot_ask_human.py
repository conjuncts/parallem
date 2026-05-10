import logging
from dotenv import load_dotenv
import parallem as pllm

load_dotenv()


def chatbot(agt: pllm.AgentContext):
    conv = agt.get_msg_state()
    while True:
        # No exit: Ctrl+C to stop.
        user_input = conv.ask_human("Send a message:")
        print(f"[USER] {user_input.final_answer}")
        resp = conv.ask_llm()
        print(f"[LLM] {resp.final_answer}")


with pllm.resume_directory(
    ".pllm/example/chatbot",
    provider="openai",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
) as orch:
    with orch.agent() as agt:
        chatbot(agt)
