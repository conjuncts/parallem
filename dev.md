From workspace root, install docs deps with uv sync --group docs (or uv sync --group dev if you want full dev tooling).
Start live docs server with uv run mkdocs serve.
Open http://127.0.0.1:8000 and edits in index.md (and other files under docs) hot-reload automatically.
If port 8000 is busy: uv run mkdocs serve -a 127.0.0.1:8001.