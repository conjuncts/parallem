from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv
import parallem as pllm
import polars as pl
from colorama import Fore, Style, init

init(autoreset=True)
load_dotenv()


# Begin conversation_uuid persistence logic

CONV_UUIDS_LOCATION = Path(".pllm/example/chatbot/conversations.parquet")


def load_conversations():
    if CONV_UUIDS_LOCATION.exists():
        df = pl.read_parquet(CONV_UUIDS_LOCATION)
    else:
        df = pl.DataFrame(schema={"conversation_uuid": pl.Utf8, "title": pl.Utf8})
    return df


def select_conversation(conv_df):
    """Have the user select a conversation."""

    print()
    print("Existing conversations")
    print("=" * 22)
    for i, row in enumerate(conv_df.iter_rows()):
        print(f'{Fore.BLUE}{i}{Style.RESET_ALL}: "{row[1]}"')
    print(f"{Fore.BLUE}{len(conv_df)}{Style.RESET_ALL}: [New conversation]")
    print(f"{Fore.BLUE}Enter{Style.RESET_ALL} to quit")
    choice = input("Select a number: ")
    if choice == "":
        return None
    choice = int(choice)
    if choice == len(conv_df):
        # generate new conversation UUID
        new_uuid = str(uuid4())
        return new_uuid
    else:
        return conv_df[choice, "conversation_uuid"]


def update_conversation(conv_df: pl.DataFrame, conv_uuid: str, title: str):
    """Update the title of an existing conversation."""
    conv_df = pl.concat(
        [
            conv_df,
            pl.DataFrame({"conversation_uuid": [conv_uuid], "title": [title]}),
        ]
    )
    conv_df = conv_df.unique(
        subset="conversation_uuid", keep="last", maintain_order=True
    )
    conv_df.write_parquet(CONV_UUIDS_LOCATION)
    return conv_df


def exit_to_quit(x):
    out = input(x)
    if out.lower() == "":
        raise KeyboardInterrupt
    return out


# Begin parallem logic


def chatbot(agt: pllm.AgentContext):
    conv = agt.get_msg_state().load()

    agt.print("Current messages:", conv.resolve())
    out = input("Send a message (enter to quit): ")
    while out:
        conv.append(out)
        conv.ask_llm()
        agt.print("Response:", conv[-1].resolve())
        out = input("Send a message (enter to quit): ")

    conv.save()
    return conv[0] if len(conv) > 0 else "[empty]"


with pllm.resume_directory(
    ".pllm/example/chatbot",
    provider="openai",
    strategy="sync",
    dashboard=True,
) as orch:
    # Demonstrate a ChatGPT-like chatbot with multiple distinct conversations.

    # Use your favorite tool (Postgres, Redis, etc.) to store conversation_uuids.
    # Polars is used for demonstration purposes.
    conv_df = load_conversations()
    conv_uuid = select_conversation(conv_df)
    if conv_uuid:
        with orch.agent(name=f"chatbot_{conv_uuid}") as agt:
            title = chatbot(agt)
            update_conversation(conv_df, conv_uuid, title)
