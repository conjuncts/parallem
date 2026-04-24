# Stress test: use GPT to count syllables for >370k words

import pathlib
import time

from dotenv import load_dotenv
from tqdm import tqdm

word_path = pathlib.Path(__file__).parent / "txts" / "words_alpha.txt"
if not word_path.exists():
    # Download the file automatically
    print("Downloading word list...")
    import requests

    url = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
    response = requests.get(url)
    word_path.parent.mkdir(parents=True, exist_ok=True)
    with open(word_path, "w") as f:
        f.write(response.text)
    print("Word list downloaded to ", word_path)

# Load words
import polars as pl  # noqa: E402

df = pl.read_csv(word_path, has_header=False, new_columns=["word"]).with_columns(
    pl.col("word").str.strip_chars()
)
df = df.unique(maintain_order=True)
df = df.head(100)
# print(df)

# Syllable count agent
import parallem as pllm  # noqa: E402


def syllable_count_agent(agt: pllm.AgentContext, word: str):
    ct = agt.ask_llm(
        f'How many syllables are in "{word}"? Only return the number, no explanation.',
        reasoning={"effort": "minimal"},  # NOTE: this is specific to openai.
        # max_completion_tokens=20,
        max_output_tokens=20,
    )
    try:
        return int(ct.final_answer.strip()), None
    except ValueError:
        return None, ct.final_answer.strip()


# Run the agent on all words
load_dotenv()
orch = pllm.resume_directory(
    ".pllm/example/stress_test",
    llm="gpt-5-nano",
    hash_by=["llm"],
    provider="openai",  # NOTE: other providers can be very expensive - be careful!
    strategy="batch",
    tweaks={
        "batch_max_size": 10000,
    },
)

collector = []
for (word,) in tqdm(df.iter_rows(), total=df.height):
    with orch.agent(f"syllable_count_{word}") as agt:
        count = syllable_count_agent(agt, word)
        collector.append((word, *count))

# Explicitly time submission
start_time = time.time()
orch.finalize_and_persist()
end_time = time.time()
print(f"Took {end_time - start_time:.2f} seconds to submit/retrieve batch.")

df = pl.DataFrame(
    collector, schema=["word", "syllable_count", "raw_output"], orient="row"
)
print(df)
df.write_parquet("syllable_counts.parquet")
