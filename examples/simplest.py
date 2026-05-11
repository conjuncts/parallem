import parallem as pllm

# Put OPENAI_API_KEY in .env file
with pllm.resume_directory(
    ".pllm/simplest",
    provider="bedrock",
    strategy="sync",
    dashboard=True,
    llm="moonshotai.kimi-k2.5",
    hash_by=["llm"],
    load_dotenv=True,
) as orch:
    with orch.agent() as agt:
        resp = agt.ask_llm("Please name a power of 3.", hash_by=["llm"])
        print(resp.final_answer)
    # print(orch.export_polars())
