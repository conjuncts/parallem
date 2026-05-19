import polars as pl
from pydantic import BaseModel
import parallem as pllm

# df = pl.read_csv('hf://datasets/hugginglearners/amazon-reviews-sentiment-analysis/amazon_reviews.csv')
df = pl.read_csv("examples/stress/txts/amazon_reviews.csv").select("reviewText").head(10)


class RatingScore(BaseModel):
    score: float


def rating_agent(agt: pllm.AgentContext, review: str) -> str:
    resp = agt.ask_llm(
        "Attached is an amazon review. What is its expected star rating out of 5?",
        review,
        reasoning={"effort": "minimal"},
        structured_output=RatingScore,
        salt=2,
    )
    return resp.final_json


with pllm.resume_directory(
    ".pllm/simplest-batch",
    provider="openai",
    strategy="sync",  # "batch" gives 50% discount
    dashboard=True,
    load_dotenv=True,
) as orch:
    collector = []
    for i, row in enumerate(df.iter_rows(named=True)):
        with orch.agent(f"agent_{i}") as agt:
            rating = rating_agent(agt, row["reviewText"])
            collector.append({**row, **rating})

result_df = pl.DataFrame(collector)
print(result_df)

# ┌─────────────────────────────────┬───────┐
# │ reviewText                      ┆ score │
# │ ---                             ┆ ---   │
# │ str                             ┆ i64   │
# ╞═════════════════════════════════╪═══════╡
# │ No issues.                      ┆ 5     │
# │ Purchased this for my device, … ┆ 4     │
# │ it works as expected. I should… ┆ 3     │
# │ This think has worked out grea… ┆ 4     │
# │ Bought it with Retail Packagin… ┆ 4     │
# │ It's mini storage.  It doesn't… ┆ 4     │
# │ I have it in my phone and it n… ┆ 4     │
# │ It's hard to believe how affor… ┆ 4     │
# │ Works in a HTC Rezound.  Was r… ┆ 4     │
# │ in my galaxy s4, super fast ca… ┆ 3     │
# └─────────────────────────────────┴───────┘
