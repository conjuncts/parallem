"""
Integration tests for batch processing functionality.

These tests validate batch strategy with various features:
- Simple text queries
- Web search tools
- Custom function tools
- Structured output (Pydantic models)
- Image inputs

The tests verify that batch files are correctly generated in JSONL format,
not that they execute (batch mode defers execution).
"""

from polars.testing import assert_frame_equal
import pytest
from pathlib import Path
from pydantic import BaseModel
from PIL import Image
from unittest.mock import Mock

import parallem as pllm
import polars as pl
from tests.integration.batch.data import (
    data_batch_full_anthropic,
    data_batch_full_google,
    data_batch_full_openai,
)


class MyModel(BaseModel):
    final_answer: str


@pytest.fixture
def sample_tools():
    """Fixture for sample function tools"""
    return [
        {
            "type": "function",
            "name": "count_files",
            "description": "Count the number of files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "The path to the directory to count files in.",
                    },
                },
                "required": ["directory"],
            },
        }
    ]


@pytest.fixture
def sample_image():
    """Fixture for sample test image"""
    image_path = Path("tests/data/images/Nokota_Horses_cropped.jpg")
    img = Image.open(image_path)
    img.thumbnail((100, 100))
    return img


def _assert_batch_file_matches_expected(
    temp_integration_dir, provider: str, expected_data: str
):
    batch_dir = temp_integration_dir / f"full_batch_{provider}" / "batch-in"
    assert batch_dir.exists(), "Batch directory not created"

    batch_files = list(batch_dir.glob("*.jsonl"))
    assert len(batch_files) == 1, f"Expected 1 batch file, found {len(batch_files)}"

    with open(batch_files[0], "r") as f:
        generated_data = f.read()
    generated_lines = generated_data.strip().split("\n")
    generated_lines = [line.strip() for line in generated_lines if line.strip()]

    expected_lines = expected_data.strip().split("\n")

    assert len(generated_lines) == 5, f"Expected 5 calls, got {len(generated_lines)}"
    assert len(expected_lines) == 5, f"Expected data has {len(expected_lines)} lines"

    mismatch_in = []
    for i, (gen, exp) in enumerate(zip(generated_lines, expected_lines)):
        if gen != exp:
            mismatch_in.append(i)

    if mismatch_in:
        test_debug_dir = Path("tests/data/diffs")
        test_debug_dir.mkdir(exist_ok=True)
        with open(test_debug_dir / f"generated_{provider}.jsonl", "w") as f:
            f.write("\n".join(generated_lines))
        with open(test_debug_dir / f"expected_{provider}.jsonl", "w") as f:
            f.write("\n".join(expected_lines))
    assert not mismatch_in, (
        f"Lines with mismatches: {mismatch_in}. "
        + f"See tests/data/diffs/generated_{provider}.jsonl and "
        + f"tests/data/diffs/expected_{provider}.jsonl for details."
    )


def _common_asks(agt, sample_tools, sample_image):
    # Test 1: Simple text query
    agt.ask_llm("Please name a power of 3.")

    # Test 2: Web search tool
    agt.ask_llm(
        "In 1 sentence, what is AAPL's current price?",
        tools=[pllm.tools.WebSearchTool()],
    )

    # Test 3: Custom function tool
    agt.ask_llm(
        "How many files are in ~/examples? Give the final answer in words.",
        tools=sample_tools,
    )

    # Test 4: Structured output
    agt.ask_llm("What is the capital of France?", structured_output=MyModel)

    # Test 5: Image input
    agt.ask_llm("What animal is this?", sample_image)


def test_full_batch_openai(temp_integration_dir, sample_tools, sample_image):
    """Test full batch file generation with all features for OpenAI"""
    provider = "openai"
    expected_data = data_batch_full_openai

    with pllm.resume_directory(
        temp_integration_dir / f"full_batch_{provider}",
        provider=provider,
        strategy="batch",
        hash_by=["llm"],
        tweaks={"batch_user_confirmation": False},
        client=False,  # Don't use real client
        save_input=True,
    ) as orch:
        # Mock the provider to prevent actual submission
        mock_submit = Mock(return_value=f"msgbatch_uuid_{provider}")
        orch._provider.submit_batch_to_provider = mock_submit

        with orch.agent() as agt:
            _common_asks(agt, sample_tools, sample_image)

    # Verify the mock was called (batch was submitted)
    assert mock_submit.called, "Batch submission was not attempted"
    _assert_batch_file_matches_expected(temp_integration_dir, provider, expected_data)

    # Verify save_input created the input tables. Only works for openai.
    inputs_dir = temp_integration_dir / f"full_batch_{provider}" / "inputs"
    assert inputs_dir.exists(), "Inputs directory not created"

    # Check history_table.parquet exists and has correct structure
    history_table_path = inputs_dir / "history_table.parquet"
    assert history_table_path.exists(), "history_table.parquet not created"

    df_history = pl.read_parquet(history_table_path)
    df_history_exp = pl.read_parquet(
        "tests/data/inputs/test_batch/history_table.parquet"
    )
    assert_frame_equal(df_history, df_history_exp)

    # Check msg_content_table.parquet exists and has correct structure
    msg_content_table_path = inputs_dir / "msg_content_table.parquet"
    assert msg_content_table_path.exists(), "msg_content_table.parquet not created"

    df_msg = pl.read_parquet(msg_content_table_path)
    df_msg_exp = pl.read_parquet(
        "tests/data/inputs/test_batch/msg_content_table.parquet"
    )
    assert_frame_equal(df_msg, df_msg_exp)


def test_full_batch_google(temp_integration_dir, sample_tools, sample_image):
    """Test full batch file generation with all features for Google"""
    provider = "google"
    expected_data = data_batch_full_google

    with pllm.resume_directory(
        temp_integration_dir / f"full_batch_{provider}",
        provider=provider,
        strategy="batch",
        hash_by=["llm"],
        tweaks={"batch_user_confirmation": False},
        client=False,  # Don't use real client
        save_input=False,
    ) as orch:
        # Mock the provider to prevent actual submission
        mock_submit = Mock(return_value=f"mock_batch_uuid_{provider}")
        orch._provider.submit_batch_to_provider = mock_submit

        with orch.agent() as agt:
            _common_asks(agt, sample_tools, sample_image)

    # Verify the mock was called (batch was submitted)
    assert mock_submit.called, "Batch submission was not attempted"
    _assert_batch_file_matches_expected(temp_integration_dir, provider, expected_data)


def test_full_batch_anthropic(temp_integration_dir, sample_tools, sample_image):
    """Test full batch file generation with all features for Anthropic"""
    provider = "anthropic"
    expected_data = data_batch_full_anthropic

    with pllm.resume_directory(
        temp_integration_dir / f"full_batch_{provider}",
        provider=provider,
        strategy="batch",
        hash_by=["llm"],
        tweaks={"batch_user_confirmation": False},
        client=False,  # Don't use real client
        save_input=False,
    ) as orch:
        mock_submit = Mock(return_value=f"mock_batch_uuid_{provider}")
        orch._provider.submit_batch_to_provider = mock_submit

        with orch.agent() as agt:
            _common_asks(agt, sample_tools, sample_image)

    assert mock_submit.called, "Batch submission was not attempted"
    _assert_batch_file_matches_expected(temp_integration_dir, provider, expected_data)
