
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
        expected = "f7649a6338429d417aad789788bfa2c268153ba554a88a6e87516d224f5f0bb3"
        assert result == expected

    def test_hash_golden_order_2(self):
        """Regression: Order-dependent hashing must remain stable"""
        result = compute_hash("Instructions", ["Document 2", "Document 1"])
        expected = "633470c100183a72eace5eca51e253f0d4d14277a38b548290a92c864dbecdcf"
        assert result == expected


    def test_hash_golden_with_image(self):
        """Regression: Image hashing must remain stable"""
        img = Image.new("RGB", (5, 5), color="blue")
        result = compute_hash("Process images", ["text", img])
        expected = "4f32d8597753e7add4272135b7a5b94bffd634384bce587f1252df9d498e2437"
        assert result == expected

    def test_hash_golden_no_instructions(self):
        """Regression: Hashing without instructions must remain stable"""
        result = compute_hash(None, ["Document 1", "Document 2"])
        expected = "207b3643bcb25e27637ad23d18973777d0054bbe30893f8fb218eba5ab7ea19d"
        assert result == expected

    def test_hash_golden_empty_documents(self):
        """Regression: Instructions-only hashing must remain stable"""
        result = compute_hash("Just instructions", [])
        expected = "0c20f4a03fe4dcf22b2c17499f3ac135bdfebd281461e7247707b7dfdc3084ee"
        assert result == expected


    def test_hash_golden_salt_v1(self):
        """Regression: Salted hashing with v1 salt must remain stable"""
        result = compute_hash("Just instructions", [], salt="v1")
        expected = "c09177c7f726faef755e378af71e9250a4ec7a6b57bf9adab340d64641e57bab"
        assert result == expected

    def test_hash_golden_salt_v2(self):
        """Regression: Salted hashing with v2 salt must remain stable"""
        result = compute_hash("Just instructions", [], salt="v2")
        expected = "9f83cb76edd76d97ec990abfac5cd428e61ebf690149eb989ff0f7bad3af2329"
        assert result == expected

    def test_hash_golden_empty_string_document(self):
        """Regression: Hashing empty string document must remain stable"""
        result = compute_hash("Test", [""])
        expected = "532eaabd9574880dbf76b9b8cc00832c20a6ec113d682299550d7a6e0f345e25"
        assert result == expected

    def test_hash_golden_no_salt_vs_salt_collision(self):
        """Regression: Verify no collision between ["ab", "cd"] and ["ab"] with salt "cd"

        This regression test ensures the fix for salt handling remains effective.
        Without proper salt handling, these two calls would produce identical hashes
        due to concatenation, causing cache collisions.
        """
        hash_no_salt = compute_hash(None, ["ab", "cd"])
        hash_with_salt = compute_hash(None, ["ab"], salt="cd")

        expected1 = "88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589"
        expected2 = "cf8595cff0931756518f438b369a9cf51f676cc948f46df00fb978a7000056ef"
        assert hash_no_salt == expected1
        assert hash_with_salt == expected2
        assert hash_no_salt != hash_with_salt
