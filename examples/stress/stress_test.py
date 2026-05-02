# Stress test: use GPT to count syllables for >370k words
import time
from tqdm import tqdm
import polars as pl

import parallem as pllm

df = pl.read_csv(
    "examples/stress/txts/words_100k.txt", has_header=False, new_columns=["word"]
).with_columns(pl.col("word").str.strip_chars())


def syllable_count_agent(agt: pllm.AgentContext, word: str):
    ct = agt.ask_llm(
        f'How many syllables are in "{word}"? Only return the number, no explanation.',
        reasoning={"effort": "minimal"},
        max_output_tokens=20,
    )
    try:
        return int(ct.final_answer.strip()), None
    except ValueError:
        return None, ct.final_answer.strip()


# Run the agent on all words
orch = pllm.resume_directory(
    ".pllm/example/fresh2/stress_test",
    llm="gpt-5-nano",
    hash_by=["llm"],
    provider="openai",
    strategy="batch",
    tweaks={
        "batch_max_size": 10000,
    },
    load_dotenv=True,
)

collector = []
for (word,) in tqdm(df.iter_rows(), total=df.height):
    with orch.agent(f"syllable_count_{word}") as agt:
        count = syllable_count_agent(agt, word)
        collector.append((word, *count))

# Time submission
start_time = time.time()
orch.finalize_and_persist()
end_time = time.time()
print(f"Took {end_time - start_time:.2f} seconds to submit/retrieve batch.")

df = pl.DataFrame(
    collector,
    schema={
        "word": pl.Utf8,
        "syllable_count": pl.Int64,
        "raw_output": pl.Utf8,
    },
    orient="row",
)
print(df)
df.write_parquet("examples/stress/txts/syllable_counts.parquet")
