import polars as pl

df = pl.read_parquet(
    ".pllm/example/stress/stress_1m_v1/datastore/apimeta/openai-responses.parquet"
)
print(df)

number_expended = df.height
number_required = 1_000_000

inp_cost = df["usage.input_tokens"].sum() / 1e6
print("Input Mtok: ", inp_cost)  # 50.2
print(f"That costed: ${inp_cost * 0.025:.2g} USD")

cost = df["usage.output_tokens"].sum() / 1e6
print("Output Mtok: ", cost)  # 488.4 /* 0.20
print(f"That costed: ${cost * 0.20:.2g} USD")

full_inp_cost = 0.025 * (inp_cost / number_expended) * number_required
full_out_cost = 0.20 * (cost / number_expended) * number_required
print("\nExtrapolated cost: ")
print(f"Input: ${full_inp_cost:.2g} USD")
print(f"Output: ${full_out_cost:.2f} USD")
print(f"Total: ${full_inp_cost + full_out_cost:.2f} USD")
