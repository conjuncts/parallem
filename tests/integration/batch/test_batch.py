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

import json
import pytest
from pathlib import Path
from pydantic import BaseModel
from PIL import Image
from unittest.mock import Mock

import parallellm as plm
from tests.integration.batch.data import data_batch_full_openai


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


def normalize_batch_line(line_dict):
    """Normalize a batch line for comparison by removing variable fields"""
    normalized = line_dict.copy()
    
    # Remove custom_id as it contains session/sequence info
    if "custom_id" in normalized:
        del normalized["custom_id"]
    
    # For OpenAI format, normalize the body
    if "body" in normalized:
        body = normalized["body"]
        # Sort tools if present for consistent comparison
        if "tools" in body and body["tools"]:
            body["tools"] = sorted(body["tools"], key=lambda x: json.dumps(x, sort_keys=True))
    
    return normalized


def test_full_batch_openai(temp_integration_dir, sample_tools, sample_image):
    """Test full batch file generation for OpenAI with all features"""
    
    with plm.resume_directory(
        temp_integration_dir / "full_batch_openai",
        provider="openai",
        strategy="batch",
        hash_by=["llm"],
        tweaks=plm.types.MinorTweaks(batch_user_confirmation=False),
    ) as orch:
        # Mock the provider to prevent actual submission to OpenAI
        mock_submit = Mock(return_value="mock_batch_uuid_123")
        orch._provider.submit_batch_to_provider = mock_submit
        
        with orch.agent() as agt:
            # Test 1: Simple text query
            agt.ask_llm("Please name a power of 3.")
            
            # Test 2: Web search tool
            agt.ask_llm(
                "In 1 sentence, what is AAPL's current price?",
                tools=[plm.tools.WebSearchTool()],
            )
            
            # Test 3: Custom function tool
            agt.ask_llm(
                "How many files are in ~/examples? Give the final answer in words.",
                tools=sample_tools,
            )
            
            # Test 4: Structured output
            agt.ask_llm(
                "What is the capital of France?",
                text_format=MyModel
            )
            
            # Test 5: Image input
            agt.ask_llm("What animal is this?", sample_image)
        
    # Verify the mock was called (batch was submitted)
    assert mock_submit.called, "Batch submission was not attempted"
    
    # Explicitly close the datastore to prevent Windows file lock issues
    # orch._backend._get_datastore().close()
    
    # Batch file should be written after agent exits
    batch_dir = temp_integration_dir / "full_batch_openai" / "batch-in"
    assert batch_dir.exists(), "Batch directory not created"
    
    batch_files = list(batch_dir.glob("*.jsonl"))
    assert len(batch_files) == 1, f"Expected 1 batch file, found {len(batch_files)}"
    
    # Read generated batch file
    with open(batch_files[0], "r") as f:
        generated_lines = [json.loads(line.strip()) for line in f if line.strip()]
    
    # Parse expected data
    expected_lines = [
        json.loads(line.strip()) 
        for line in data_batch_full_openai.strip().split("\n") 
        if line.strip()
    ]
    
    assert len(generated_lines) == 5, f"Expected 5 calls, got {len(generated_lines)}"
    assert len(expected_lines) == 5, f"Expected data has {len(expected_lines)} lines"
    
    # Compare each line (excluding custom_id which varies)
    for i, (gen, exp) in enumerate(zip(generated_lines, expected_lines)):
        gen_norm = normalize_batch_line(gen)
        exp_norm = normalize_batch_line(exp)
        
        # Check structure matches
        assert gen_norm.keys() == exp_norm.keys(), \
            f"Line {i}: Key mismatch. Generated: {gen_norm.keys()}, Expected: {exp_norm.keys()}"
        
        # Check method and url
        assert gen_norm.get("method") == exp_norm.get("method"), \
            f"Line {i}: Method mismatch"
        assert gen_norm.get("url") == exp_norm.get("url"), \
            f"Line {i}: URL mismatch"
        
        # Check body structure
        if "body" in gen_norm and "body" in exp_norm:
            gen_body = gen_norm["body"]
            exp_body = exp_norm["body"]
            
            # Check model
            assert gen_body.get("model") == exp_body.get("model"), \
                f"Line {i}: Model mismatch"
            
            # Check input (first message content)
            if "input" in gen_body and "input" in exp_body:
                gen_first_msg = gen_body["input"][0] if gen_body["input"] else {}
                exp_first_msg = exp_body["input"][0] if exp_body["input"] else {}
                
                if isinstance(gen_first_msg.get("content"), str):
                    assert gen_first_msg.get("content") == exp_first_msg.get("content"), \
                        f"Line {i}: Input content mismatch"
            
            # Check tools presence
            assert ("tools" in gen_body) == ("tools" in exp_body), \
                f"Line {i}: Tools presence mismatch"
            
            # If tools present, check tool names
            if "tools" in gen_body and gen_body["tools"]:
                gen_tool_names = {t.get("name") or t.get("type") for t in gen_body["tools"]}
                exp_tool_names = {t.get("name") or t.get("type") for t in exp_body["tools"]}
                assert gen_tool_names == exp_tool_names, \
                    f"Line {i}: Tool names mismatch. Generated: {gen_tool_names}, Expected: {exp_tool_names}"
