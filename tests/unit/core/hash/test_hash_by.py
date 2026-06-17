"""Regression tests for hash_by functionality.

Tests verify that different hash_by options produce correct and distinct hash values.
This ensures that cache differentiation works correctly when using different hash_by
configurations.
"""

from pydantic import BaseModel
import pytest

from parallem.core.agent.agent import AgentContext
from parallem.types import LLMIdentity
from parallem.tools.server import WebSearchTool


@pytest.fixture
def params():
    """Fixture for common parameters used in tests."""
    return {
        "instructions": "Test instructions",
        "strict_documents": ["Doc 1", "Doc 2"],
        "llm": LLMIdentity("gpt-4o-mini", provider_type="openai"),
        "structured_output": None,
        "tools": None,
    }


class TestHashByToolNames:
    """Tests for hash_by=['tool_names'] functionality."""

    def test_hash_by_tool_names_with_tools(self, params):
        """Test that tool names are included in the hash."""
        params["tools"] = [{"name": "tool1"}, {"name": "tool2"}]
        hash_result, terms = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["tool_names"],
            provider_type="openai",
        )

        assert hash_result == "23cdd9802923d8b6ec7a27aee3f4e49231ea1b7e5d202a78d82cb40678a60457"
        assert len(terms) == 1
        assert terms[0] == '["tool1", "tool2"]'

    def test_hash_by_tool_names_without_tools(self, params):
        """Test that no tool names are included when tools are None."""
        hash_result, terms = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["tool_names"],
            provider_type="openai",
        )

        assert hash_result == "4c61214b1872355498ce8424867b2bf2faa0956843423a36f0fe99063e7abff6"
        assert len(terms) == 0

    def test_hash_by_tool_names_different_order(self, params):
        """Test that different tool name order produces different hash."""
        params["tools"] = [{"name": "tool2"}, {"name": "tool1"}]

        hash_result, terms = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["tool_names"],
            provider_type="openai",
        )

        assert hash_result == "33787a33a4029e1ee5edd1ff3ae0f962408b3ffc2ebeb3b4d35dae68848a27ba"
        assert len(terms) == 1
        assert terms[0] == '["tool2", "tool1"]'

    def test_ask_llm_hash_by_tool_names(self, params):
        params["tools"] = [WebSearchTool()]
        hash4, terms = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["tool_names"],
            provider_type="test_provider",
        )
        assert hash4 == "0d8c15ee81c4303b97fc1ae90691700484cc50f7c0e041e776cb801ac4774f29"
        assert len(terms) == 1
        assert terms[0] == '["web_search"]'


class TestHashByStructuredOutput:
    """Tests for hash_by=['structured_output'] functionality."""

    def test_hash_by_structured_output_with_schema(self, params):
        """Test that structured output schema is included in the hash."""

        class OutputSchema(BaseModel):
            name: str
            age: int

        params["structured_output"] = OutputSchema

        hash_result, terms = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["structured_output"],
            provider_type="openai",
        )

        assert hash_result == "96aa46e695fb9591eaf12f0f94edf3ae0de2f907961bb02fb0ec696791390e94"
        assert len(terms) == 1
        assert "properties" in terms[0]
        assert "OutputSchema" in terms[0]

    def test_hash_by_structured_output_different_schema(self, params):
        """Test that different schemas produce different hashes."""

        class OutputSchema(BaseModel):
            title: str
            description: str

        params["structured_output"] = OutputSchema
        hash_result, terms = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["structured_output"],
            provider_type="openai",
        )

        assert hash_result == "528f5bc76a79b04f3bd97f28ffa73a6c43a2a6c7556b4a751451920f468c72af"
        assert len(terms) == 1
        assert "OutputSchema" in terms[0]

    def test_hash_by_structured_output_without_schema(self, params):
        """Test that no schema is included when structured_output is None."""
        hash_result, terms = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["structured_output"],
            provider_type="openai",
        )

        assert hash_result == "4c61214b1872355498ce8424867b2bf2faa0956843423a36f0fe99063e7abff6"
        assert len(terms) == 0


class TestHashByKwargs:
    """Tests for hash_by=['kwargs'] functionality."""

    def test_hash_by_kwargs_with_values(self, params):
        """Test that kwargs are included in the hash."""
        hash_result, terms = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["kwargs"],
            provider_type="openai",
            kwargs={"temperature": 0.7, "max_tokens": 100},
        )

        assert hash_result == "35d9cb31900d28473a63f8e8cf35966bf88d8e389f6784d74ebba05c9ce96f55"
        assert len(terms) == 1
        assert "temperature" in terms[0]
        assert "0.7" in terms[0]

    def test_hash_by_kwargs_without_values(self, params):
        """Test that no kwargs are included when kwargs is None."""
        hash_result, terms = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["kwargs"],
            provider_type="openai",
            kwargs=None,
        )

        assert hash_result == "4c61214b1872355498ce8424867b2bf2faa0956843423a36f0fe99063e7abff6"
        assert len(terms) == 0

    def test_hash_by_kwargs_different_values(self, params):
        """Test that different kwargs produce different hashes."""
        hash_result, terms = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["kwargs"],
            provider_type="openai",
            kwargs={"temperature": 0.5, "max_tokens": 200},
        )

        assert hash_result == "2df861d75a75e495012803f7d52b077104bcd25a5bd5a8f2c0f7981896b7f9c1"
        assert len(terms) == 1
        assert "temperature" in terms[0]
        assert "0.5" in terms[0]


class TestHashByAll:
    """Tests for hash_by=['all'] functionality."""

    def test_hash_by_all_with_all_options(self, params):
        """Test that all options are included when hash_by=['all']."""

        class OutputSchema(BaseModel):
            name: str
            age: int

        params["structured_output"] = OutputSchema
        params["tools"] = [{"name": "tool1"}, {"name": "tool2"}]
        hash_result, terms = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["all"],
            provider_type="openai",
            kwargs={"temperature": 0.7},
        )

        assert hash_result == "6867f8b487427fb8307bc7f7ca895ec785ddf6b3ef3360a1d063724c087a315a"


class TestHashByCombinations:
    """Tests for combinations of hash_by options."""

    def test_hash_by_multiple_options(self, params):
        """Test combining multiple hash_by options."""

        class OutputSchema(BaseModel):
            result: str

        params["structured_output"] = OutputSchema
        params["tools"] = [{"name": "search"}]

        # Test with multiple options
        hash_result, terms = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["tool_names", "kwargs"],
            provider_type="openai",
            kwargs={"temperature": 0.8},
        )

        assert len(terms) == 2
        assert any("search" in term for term in terms)
        assert any("temperature" in term for term in terms)

    def test_hash_by_tools_vs_tool_names(self, params):
        """Test that hash_by=['tools'] and hash_by=['tool_names'] differ."""
        params["tools"] = [{"name": "tool1", "description": "Tool 1"}]

        hash_tools, _ = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["tools"],
            provider_type="openai",
        )

        hash_tool_names, _ = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["tool_names"],
            provider_type="openai",
        )

        # Adding description to tool changes the full tools hash
        # but not the tool_names hash
        assert hash_tools != hash_tool_names


@pytest.mark.skip(reason="hash_by tools not ready yet")
class TestExistingHashByTools:
    """Regression tests for existing hash_by=['tools'] functionality."""

    def test_ask_llm_hash_by_tools(self):
        params = {
            "instructions": "Test instructions",
            "strict_documents": ["Doc 1", "Doc 2"],
            "llm": None,
            "structured_output": None,
            "tools": [{"name": "tool1"}, {"name": "tool2"}],
        }
        hash1, _ = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["tools"],
            provider_type="test_provider",
        )
        assert hash1 == "ffd6b7504c0ae7215320b8131a4b33a2e08630148ab50cd1a15503f4bae84ad9"
        params["tools"] = None
        hash2, _ = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["tools"],
            provider_type="test_provider",
        )
        assert hash2 == "31644a7cb90c010b62f2566223cad9c06ecc6fe238fa65ac309b2bec2d71805e"
        params["tools"] = [{"name": "tool2"}, {"name": "tool1"}]
        hash3, _ = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["tools"],
            provider_type="test_provider",
        )
        assert hash3 == "0634d882f06f350e9ab77272ce7d10119369a2acfd1dcb09042aead448a38d6e"
        params["tools"] = [WebSearchTool()]
        hash4, _ = AgentContext._compute_hash(
            None,
            params=params,
            salt=None,
            hash_by=["tools"],
            provider_type="test_provider",
        )
        assert hash4 == "3038c81daa35cbd3aef8949d6cc3196754a6957c342f8a08748d54cf5c259534"
