from PIL import Image
import pytest

from parallem.core.hash import compute_hash, serialize_tools_for_hash
from parallem.tools.server import WebSearchTool


class TestComputeHash:
    """Test the compute_hash function"""

    def test_hash_unsupported_type(self):
        """Test error handling for unsupported document types"""
        instructions = "Test"
        documents = ["valid", 123]  # Invalid type

        with pytest.raises(ValueError, match="Unsupported document type"):
            compute_hash(instructions, documents)

    def test_serialize_tools_is_deterministic(self):
        """Tool serialization should ignore dict ordering inside tool kwargs."""
        tool1 = WebSearchTool(kwargs={"b": 2, "a": 1})
        tool2 = WebSearchTool(kwargs={"a": 1, "b": 2})

        serialized1 = serialize_tools_for_hash([tool1])
        serialized2 = serialize_tools_for_hash([tool2])

        assert serialized1 == serialized2
        assert '"server_tool_type":"web_search"' in serialized1


class TestHashRegressions:
    """Regression tests that verify hash values remain stable."""

    def test_hash_golden_text_only(self):
        """Regression: Text-only hashing must remain stable"""
        result = compute_hash("Instructions", ["Document 1", "Document 2"])
        expected = "ffcfcf48884dd651ab7dab48ed4c1ff11552a56c9281c2ff9b5ac7b287155fb9"
        assert result == expected

    def test_hash_golden_order_2(self):
        """Regression: Order-dependent hashing must remain stable"""
        result = compute_hash("Instructions", ["Document 2", "Document 1"])
        expected = "623974c1f8e5655218e44a90c6960d084e13236d5daa10ed19ea757971a6571c"
        assert result == expected

    def test_hash_golden_with_image(self):
        """Regression: Image hashing must remain stable"""
        img = Image.new("RGB", (5, 5), color="blue")
        result = compute_hash("Process images", ["text", img])
        expected = "49da99dea95d779f5138468f2e9ccd628e6b2677bfe8a57149bdaa5353a68915"
        assert result == expected

    def test_hash_golden_no_instructions(self):
        """Regression: Hashing without instructions must remain stable"""
        result = compute_hash(None, ["Document 1", "Document 2"])
        expected = "aee70174b9a105f5c346a95fa43e35473093e0c03f9092282e5fe27be8aac26f"
        assert result == expected

    def test_hash_golden_empty_documents(self):
        """Regression: Instructions-only hashing must remain stable"""
        result = compute_hash("Just instructions", [])
        expected = "a91d5e04d008d88a8575b150ed7807f94cef78a14cd00586625320aecb3ca52d"
        assert result == expected

    def test_hash_golden_salt_v1(self):
        """Regression: Salted hashing with v1 salt must remain stable"""
        result = compute_hash("Just instructions", [], salt="v1")
        expected = "77754bbf1ee12308063ef91df896fbb308b4ee8b0c9f76a01a3f3d6200a7f013"
        assert result == expected

    def test_hash_golden_salt_v2(self):
        """Regression: Salted hashing with v2 salt must remain stable"""
        result = compute_hash("Just instructions", [], salt="v2")
        expected = "325ae1cf8f6be0e38e8ed49bd5fd14a32b6607587d1d78399f63ea4eabe22992"
        assert result == expected

    def test_hash_golden_empty_string_document(self):
        """Regression: Hashing empty string document must remain stable"""
        result = compute_hash("Test", [""])
        expected = "dd30af31ec866a1a57aad4ca7349641fdb6a6034c18f879038867f31dfb0703d"
        assert result == expected

    def test_hash_golden_no_salt_vs_salt_collision(self):
        """Regression: Verify no collision between ["ab", "cd"] and ["ab"] with salt "cd"

        This regression test ensures the fix for salt handling remains effective.
        Without proper salt handling, these two calls would produce identical hashes
        due to concatenation, causing cache collisions.
        """
        hash_no_salt = compute_hash(None, ["ab", "cd"])
        hash_with_salt = compute_hash(None, ["ab"], salt="cd")

        expected1 = "bbe739600221e7d5346bd201049d90ad242fab90e1160621558bc69e1f5920ef"
        expected2 = "61d5c167ed9224e1a6fa959994dea6f481a295ce77c5c5fbc733a984e77afaf6"
        assert hash_no_salt == expected1
        assert hash_with_salt == expected2
        assert hash_no_salt != hash_with_salt
