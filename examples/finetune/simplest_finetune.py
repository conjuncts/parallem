import logging
from dotenv import load_dotenv
import parallem as pllm
import polars as pl

load_dotenv()

with pllm.resume_directory(
    ".pllm/simple/finetune",
    provider="openai",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
    # ignore_cache=True,
) as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm(
            "Please name a power of 19.",
            tag="power-of-n",
            save_input=True,
        )

        agt.print(resp.resolve())


df = pl.read_parquet(".pllm/simple/finetune/inputs/history_table.parquet")
print(df)

# shape: (1, 4)
# ┌─────────────────────────────────┬──────────────┬─────────────────────────────────┬────────────┐
# │ doc_hash                        ┆ instructions ┆ msg_hashes                      ┆ salt_terms │
# │ ---                             ┆ ---          ┆ ---                             ┆ ---        │
# │ str                             ┆ str          ┆ list[str]                       ┆ list[str]  │
# ╞═════════════════════════════════╪══════════════╪═════════════════════════════════╪════════════╡
# │ c54981bfdba02496d5b34d5c75647a… ┆ null         ┆ ["c54981bfdba02496d5b34d5c7564… ┆ []         │
# └─────────────────────────────────┴──────────────┴─────────────────────────────────┴────────────┘

df2 = pl.read_parquet(".pllm/simple/finetune/inputs/msg_content_table.parquet")
print(df2)

# shape: (1, 4)
# ┌─────────────────────────────────┬─────────────────────────────────┬──────────┬───────────┐
# │ msg_hash                        ┆ msg_value                       ┆ msg_type ┆ msg_extra │
# │ ---                             ┆ ---                             ┆ ---      ┆ ---       │
# │ str                             ┆ binary                          ┆ str      ┆ str       │
# ╞═════════════════════════════════╪═════════════════════════════════╪══════════╪═══════════╡
# │ c54981bfdba02496d5b34d5c75647a… ┆ b"Please\x20name\x20a\x20power… ┆ text     ┆ null      │
# └─────────────────────────────────┴─────────────────────────────────┴──────────┴───────────┘
