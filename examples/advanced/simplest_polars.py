import logging
import parallem as pllm
import polars as pl


def molar_mass_agent(agt: pllm.AgentContext, compound: str) -> str:
    resp = agt.ask_llm(
        f"What is the molar mass of {compound}? Just give the number in g/mol.",
        tools=[pllm.tools.WebSearchTool()],
    )
    return resp.final_answer


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
    load_dotenv=True,
    hash_by=["llm"],
) as orch:
    collector = []
    for i, row in enumerate(df.iter_rows(named=True)):
        with orch.agent(f"agent_{i}") as agt:
            mm = molar_mass_agent(agt, row["compound"])
            collector.append({**row, "molar_mass": mm})

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
