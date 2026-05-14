import gzip

import polars as pl
import json

collector = []
loc = ".pllm/stress/fresh3/stress_test/datastore/apimeta/openai-metadata.tsv.gz"

with gzip.open(loc, "rt", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        resp_id, metadata_txt = line.split("\t", 1)
        if not line:
            continue
        collector.append({
            "response_id": resp_id,
            **json.loads(metadata_txt),
        })

df = pl.json_normalize(collector)
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
