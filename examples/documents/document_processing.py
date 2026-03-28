import os
from pathlib import Path
from dotenv import load_dotenv
import requests
import parallem as pllm


documents = [
    "https://www.gutenberg.org/files/43/43-0.txt",  # Jekyll and Hyde
    "https://www.gutenberg.org/files/1952/1952-0.txt",  # Yellow Wallpaper
    "https://www.gutenberg.org/files/208/208-0.txt",  # Daisy Miller
    "https://www.gutenberg.org/files/2641/2641-0.txt",  # A Room with a View
    "https://www.gutenberg.org/files/64317/64317-0.txt",  # The Great Gatsby
]


def download_documents():
    """Download documents to examples/documents/txts."""
    txts_dir = Path("examples/documents/txts")
    txts_dir.mkdir(parents=True, exist_ok=True)
    for url in documents:
        filename = url.split("/")[-1]
        filepath = txts_dir / filename
        if not filepath.exists():
            print(f"Downloading {url} to {filepath}...")
            response = requests.get(url)
            with filepath.open("w", encoding="utf-8") as f:
                f.write(response.text)
        else:
            print(f"{filepath} already exists, skipping download.")


def summarizer_agent(agt: pllm.AgentContext, doc: str):
    resp = agt.ask_llm(
        "In 1 paragraph, how does the main character change in this literary work?", doc
    )
    return resp


if __name__ == "__main__":
    download_documents()
    load_dotenv()
    with pllm.resume_directory(
        ".pllm/example/document_processing",
        strategy="batch",
    ) as orch:
        for fname in os.listdir("examples/documents/txts"):
            with open(f"examples/documents/txts/{fname}", "r", encoding="utf-8") as f:
                doc = f.read()
            with orch.agent(fname) as agt:
                out = summarizer_agent(agt, doc)
                print(out.final_answer[:40] + "...")
