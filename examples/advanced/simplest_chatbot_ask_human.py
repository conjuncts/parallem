import logging
from dotenv import load_dotenv
import pipelinellm as pllm
from pipelinellm.core.agent.agent import AgentContext

load_dotenv()


def chatbot(agt: AgentContext):
    conv = agt.get_msg_state()
    while True:
        # No exit: Ctrl+C to stop.
        user_input = conv.ask_human("Send a message:")
        agt.print(f"[USER] {user_input.final_answer}")
        resp = conv.ask_llm()
        agt.print(f"[LLM] {resp.final_answer}")


with pllm.resume_directory(
    ".pllm/example/chatbot",
    provider="openai",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
) as orch:
    with orch.agent() as agt:
        chatbot(agt)
