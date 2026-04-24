import polars as pl

df = pl.read_parquet(
    ".pllm/example/stress_test/datastore/apimeta/openai-responses.parquet"
)
print(df)

cost = df["usage.output_tokens"].sum() / 1e6
print("Mtok: ", cost)  # 488.4 /* 0.20
print("Expected cost: ", cost * 0.20, "USD")
