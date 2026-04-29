
# Enzyme Design Arena (Usecase)

This usecase demonstrates a single Python control flow for enzyme-design tasks built on top of ParaLLeM.

- One entrypoint: `run_arena.py`
- One strategy switch: `--strategy` (`sync`, `concurrent`, `batch`)
- No core changes required

## User Config (Recommended)

Use a private per-user config at:

- `usecases/enzyme_design_arena/user/config.yaml` (ignored by git)
- Start from `usecases/enzyme_design_arena/user/config.example.yaml`

Priority order for runtime values:

1. CLI args
2. `config.yaml`
3. hardcoded defaults

For OpenAI key:

- Preferred: export `OPENAI_API_KEY`
- Optional local fallback: set `openai.api_key` in private `config.yaml`
- If both exist, `openai.prefer_config_key: true` makes `config.yaml` override env var

Parallem runtime knobs exposed by CLI/config:

- `provider` (`openai` / `google` / `anthropic` / `multi`)
- `strategy` (`sync` / `async` / `concurrent` / `batch`), where `async` maps to `concurrent`
- `llm`
- `dashboard`
- `hash_by`
- `total_job_budget` (grid + llm_extra total; default 100)
- `enable_llm_extra` (keep grid baseline, add LLM-proposed candidates)
