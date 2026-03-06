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

import parallellm as pllm
from tests.integration.batch.data import data_batch_full_openai, data_batch_full_google


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


@pytest.mark.parametrize(
    "provider,expected_data",
    [
        ("openai", data_batch_full_openai),
        ("google", data_batch_full_google),
    ],
)
def test_full_batch(
    temp_integration_dir, sample_tools, sample_image, provider, expected_data
):
    """Test full batch file generation with all features for different providers"""

    with pllm.resume_directory(
        temp_integration_dir / f"full_batch_{provider}",
        provider=provider,
        strategy="batch",
        hash_by=["llm"],
        tweaks=pllm.types.MinorTweaks(batch_user_confirmation=False),
        client=False,  # Don't use real client
    ) as orch:
        # Mock the provider to prevent actual submission
        mock_submit = Mock(return_value=f"mock_batch_uuid_{provider}")
        orch._provider.submit_batch_to_provider = mock_submit

        with orch.agent() as agt:
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
            agt.ask_llm("What is the capital of France?", text_format=MyModel)

            # Test 5: Image input
            agt.ask_llm("What animal is this?", sample_image)

    # Verify the mock was called (batch was submitted)
    assert mock_submit.called, "Batch submission was not attempted"

    # Batch file should be written after agent exits
    batch_dir = temp_integration_dir / f"full_batch_{provider}" / "batch-in"
    assert batch_dir.exists(), "Batch directory not created"

    batch_files = list(batch_dir.glob("*.jsonl"))
    assert len(batch_files) == 1, f"Expected 1 batch file, found {len(batch_files)}"

    # Read generated batch file
    with open(batch_files[0], "r") as f:
        generated_data = f.read()
    generated_lines = generated_data.strip().split("\n")
    generated_lines = [line.strip() for line in generated_lines if line.strip()]

    expected_lines = expected_data.strip().split("\n")

    assert len(generated_lines) == 5, f"Expected 5 calls, got {len(generated_lines)}"
    assert len(expected_lines) == 5, f"Expected data has {len(expected_lines)} lines"

    # Compare each line (excluding custom_id/key which varies)
    mismatch_in = []
    for i, (gen, exp) in enumerate(zip(generated_lines, expected_lines)):
        if gen != exp:
            # Write files that can be diff'd
            mismatch_in.append(i)

    if mismatch_in:
        # dump
        test_debug_dir = Path("tests/data/diffs")
        test_debug_dir.mkdir(exist_ok=True)
        with open(test_debug_dir / f"generated_{provider}.jsonl", "w") as f:
            f.write("\n".join(generated_lines))
        with open(test_debug_dir / f"expected_{provider}.jsonl", "w") as f:
            f.write("\n".join(expected_lines))
    assert not mismatch_in, (
        f"Lines with mismatches: {mismatch_in}. See tests/data/diffs/generated_{provider}.jsonl"
    )
