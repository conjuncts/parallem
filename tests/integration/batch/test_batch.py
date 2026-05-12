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

import pytest
from pathlib import Path
from pydantic import BaseModel
from PIL import Image
from unittest.mock import Mock
import zipfile

import parallem as pllm
from parallem.utils._quick_structured import _anthropic_transform_schema
from tests.integration.batch.data import (
    data_batch_full_anthropic,
    data_batch_full_google,
    data_batch_full_openai,
    data_batch_full_openai_chat,
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

    batch_files = list(batch_dir.glob("*.zip"))
    assert len(batch_files) == 1, f"Expected 1 batch zip file, found {len(batch_files)}"

    with zipfile.ZipFile(batch_files[0], "r") as z:
        jsonl_names = [n for n in z.namelist() if n.endswith(".jsonl")]
        assert len(jsonl_names) == 1, (
            f"Expected 1 JSONL inside zip, found {len(jsonl_names)}"
        )
        with z.open(jsonl_names[0], "r") as f:
            generated_data = f.read().decode("utf-8")
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


def _common_asks(agt: pllm.AgentContext, sample_tools, sample_image, _do_web_search=True):
    # Test 1: Simple text query
    agt.ask_llm(
        "Please name a power of 3.",
        instructions="No explanations needed.",
    )

    # Test 2: Web search tool
    agt.ask_llm(
        "In 1 sentence, what is AAPL's current price?",
        tools=[pllm.tools.WebSearchTool()] if _do_web_search else None,
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

    # Verify save_input created the input storage outputs.
    inputs_dir = temp_integration_dir / f"full_batch_{provider}" / "inputs"
    assert inputs_dir.exists(), "Inputs directory not created"

    multimedia_dir = inputs_dir / "multimedia"
    assert multimedia_dir.exists(), "Multimedia inputs directory not created"

    text_table_path = multimedia_dir / "text.parquet"
    assert text_table_path.exists(), "text.parquet not created"

    images_table_path = multimedia_dir / "images.parquet"
    assert images_table_path.exists(), "images.parquet not created"

    config_dir = inputs_dir / "config"
    assert config_dir.exists(), "Config inputs directory not created"
    config_zips = list(config_dir.glob("session_*.zip"))
    assert config_zips, "Session config zip not created"


def test_full_batch_openai_chat(temp_integration_dir, sample_tools, sample_image):
    """Test full batch file generation with chat completions for OpenAI"""
    provider = "openai-chat"
    expected_data = data_batch_full_openai_chat

    with pllm.resume_directory(
        temp_integration_dir / f"full_batch_{provider}",
        provider=provider,
        strategy="batch",
        hash_by=["llm"],
        tweaks={"batch_user_confirmation": False},
        client=False,  # Don't use real client
        save_input=False,
    ) as orch:
        mock_submit = Mock(return_value=f"msgbatch_uuid_{provider}")
        orch._provider.submit_batch_to_provider = mock_submit

        with orch.agent() as agt:
            # ChatCompletions does not support web search
            _common_asks(agt, sample_tools, sample_image, _do_web_search=False)

    assert mock_submit.called, "Batch submission was not attempted"
    _assert_batch_file_matches_expected(temp_integration_dir, provider, expected_data)


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


def test_full_batch_anthropic(monkeypatch, temp_integration_dir, sample_tools, sample_image):
    """Test full batch file generation with all features for Anthropic"""
    provider = "anthropic"
    expected_data = data_batch_full_anthropic

    from parallem.provider.anthropic import sdk as anthropic_sdk
    monkeypatch.setattr(
        anthropic_sdk,
        "_transform_schema",
        _anthropic_transform_schema,
        raising=True,
    )

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
