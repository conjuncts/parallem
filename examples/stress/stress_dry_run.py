import logging
import shutil
from pathlib import Path

from tqdm import tqdm

import parallem as pllm
from parallem.testing.simple_mock import mock_openai_client


def main() -> None:
    """Run stress benchmark and print throughput for cold and warm cache."""
    count = 100_000
    directory = Path(".pllm/working/stress_dry")
    keep_cache = True

    if not keep_cache and directory.exists():
        shutil.rmtree(directory)

    mock_client = mock_openai_client()

    with pllm.resume_directory(
        directory,
        provider="openai",
        strategy="sync",
        client=mock_client,
        log_level=logging.WARNING,
        dashboard=False,
    ) as orch:
        collector = []
        for i in tqdm(range(count)):
            with orch.agent(f"stress_{i}") as agt:
                resp = agt.ask_llm(f"Dummy prompt {i}")
                collector.append(resp.final_answer)
    # First time: [04:51<00:00, 343.27it/s]
    # Storage: 59 MB

    with pllm.resume_directory(
        directory,
        provider="openai",
        strategy="sync",
        client=mock_client,
        log_level=logging.WARNING,
        dashboard=False,
    ) as orch:
        collector = []
        for i in tqdm(range(100_000)):
            with orch.agent(f"stress_{i}") as agt:
                resp = agt.ask_llm(f"Dummy prompt {i}")
                collector.append(resp.final_answer)
    # Cache: [00:08<00:00, 12396.79it/s]


if __name__ == "__main__":
    main()
