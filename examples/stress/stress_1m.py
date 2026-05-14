# Stress test: 1 million

from tqdm import tqdm

import parallem as pllm
import polars as pl
from datasets import load_dataset

ds = load_dataset("Skelebor/book_titles_and_descriptions_en_clean", split="train")
print(ds.num_rows) # 1,032,335
# NOTE Cost: estimated $5

def genre_agent(agt: pllm.AgentContext, title: str):
    ct = agt.ask_llm(
        "Based on title alone, guess this book's genre. Return just the genre, no explanation needed.",
        title,
        reasoning={"effort": "minimal"},
        max_output_tokens=20,
    )
    return ct.final_answer.strip()

collector = []
with pllm.resume_directory(
    ".pllm/stress/stress_1m_v1",
    llm="gpt-5-nano",
    load_dotenv=True,
    strategy="batch",
    tweaks={
        "batch_max_size": 10000,
    },
) as orch:
    for i, example in tqdm(enumerate(ds), total=len(ds)):
        with orch.agent(i) as agt:
            out = genre_agent(agt, example["title"])
            collector.append((example["title"], out))
print(pl.DataFrame(collector, schema={"title": pl.Utf8, "genre": pl.Utf8}, orient="row"))
# 