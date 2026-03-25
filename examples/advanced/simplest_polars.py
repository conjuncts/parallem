import logging
from dotenv import load_dotenv
import parallem as pllm
import polars as pl

load_dotenv()


df = pl.DataFrame(
    ["CO2", "toluene", "methane", "CaCO3", "Pb(C2H5)4"],
    schema=["compound"],
)
with pllm.resume_directory(
    ".pllm/simplest-batch",
    provider="openai",
    strategy="sync",
    log_level=logging.DEBUG,
    dashboard=True,
    hash_by=["llm"],
    # ignore_cache=True,
) as orch:
    collector = []
    for row in df.iter_rows(named=True):
        with orch.agent() as agt:
            resp = agt.ask_llm(
                f"What is the molar mass of {row['compound']}? Just give the number in g/mol.",
                tools=[pllm.tools.WebSearchTool()],
            )

            collector.append({**row, "molar_mass": resp.final_answer})

result_df = pl.DataFrame(collector)
print(result_df)

# ┌───────────┬────────────┐
# │ compound  ┆ molar_mass │
# │ ---       ┆ ---        │
# │ str       ┆ str        │
# ╞═══════════╪════════════╡
# │ CO2       ┆ 44.01      │
# │ toluene   ┆ 92.14      │
# │ methane   ┆ 16.04      │
# │ CaCO3     ┆ 100.09     │
# │ Pb(C2H5)4 ┆ 323.45     │
# └───────────┴────────────┘
