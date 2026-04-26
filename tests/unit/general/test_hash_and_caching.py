"""
Unit tests for hash computation and caching logic

Tests the core hashing and caching functionality including:
- compute_hash function for different document types
- Hash status tracking and dashboard logging
- Cache retrieval logic in backends
- Call identifier matching
"""

import pytest
from unittest.mock import patch
from PIL import Image
from parallem.core.hash import compute_hash, serialize_tools_for_hash
from parallem.logging.dash_logger import DashboardLogger, HashStatus
from parallem.core.calls import _call_matches
from parallem.tools.server import WebSearchTool
from parallem.types import CallIdentifier


class TestComputeHash:
    """Test the compute_hash function"""

    def test_hash_deterministic_with_images(self):
        """Test that hash is deterministic with identical images"""
        instructions = "Process images"

        img1 = Image.new("RGB", (5, 5), color="blue")
        img2 = Image.new("RGB", (5, 5), color="blue")  # Identical image
        img3 = Image.new("RGB", (5, 5), color="red")  # Different image

        result1 = compute_hash(instructions, ["text", img1])
        result2 = compute_hash(instructions, ["text", img2])
        result3 = compute_hash(instructions, ["text", img3])

        assert isinstance(result1, str)
        assert len(result1) == 64
        assert result1 == result2
        assert result1 != result3

    def test_hash_unsupported_type(self):
        """Test error handling for unsupported document types"""
        instructions = "Test"
        documents = ["valid", 123]  # Invalid type

        with pytest.raises(ValueError, match="Unsupported document type"):
            compute_hash(instructions, documents)

    def test_salt_changes_hash(self):
        """Providing a salt must change the hash."""
        h1 = compute_hash(None, ["hello"])
        h2 = compute_hash(None, ["hello"], salt="v1")
        h3 = compute_hash(None, ["hello"], salt="v2")
        assert h1 != h2
        assert h1 != h3
        assert h2 != h3

    def test_salt_is_deterministic(self):
        """Same inputs with the same salt must always return the same hash."""
        h1 = compute_hash("instr", ["doc"], salt="s1")
        h2 = compute_hash("instr", ["doc"], salt="s1")
        assert h1 == h2

    def test_serialize_tools_is_deterministic(self):
        """Tool serialization should ignore dict ordering inside tool kwargs."""
        tool1 = WebSearchTool(kwargs={"b": 2, "a": 1})
        tool2 = WebSearchTool(kwargs={"a": 1, "b": 2})

        serialized1 = serialize_tools_for_hash([tool1])
        serialized2 = serialize_tools_for_hash([tool2])

        assert serialized1 == serialized2
        assert '"server_tool_type":"web_search"' in serialized1


@pytest.mark.skip("Not very informative")
class TestDashboardLogger:
    """Test DashboardLogger functionality"""

    def test_update_hash_new_entry(self):
        """Test adding a new hash entry"""
        logger = DashboardLogger(k=3, display=False)
        full_hash = "abcdef123456" + "0" * 52

        logger.update_hash(full_hash, HashStatus.SENT)

        assert len(logger._hashes) == 1
        assert "abcdef12" in logger._hashes
        assert logger._hashes["abcdef12"].status == HashStatus.SENT

    def test_update_hash_existing_entry(self):
        """Test updating an existing hash entry"""
        logger = DashboardLogger(k=3, display=False)
        full_hash = "abcdef123456" + "0" * 52

        # Add initial entry
        logger.update_hash(full_hash, HashStatus.SENT)

        # Update existing entry
        logger.update_hash(full_hash, HashStatus.RECEIVED)

        assert len(logger._hashes) == 1
        assert logger._hashes["abcdef12"].status == HashStatus.RECEIVED

    def test_hash_limit_enforcement(self):
        """Test that logger respects k limit"""
        logger = DashboardLogger(k=2, display=False)

        # Add 3 hashes (should only keep 2)
        hashes = [
            "aaaaaaaa" + "0" * 56,
            "bbbbbbbb" + "1" * 56,
            "cccccccc" + "2" * 56,
        ]

        for i, h in enumerate(hashes):
            logger.update_hash(h, HashStatus.SENT)

        assert len(logger._hashes) == 2
        # Should keep the most recent 2
        assert "bbbbbbbb" in logger._hashes
        assert "cccccccc" in logger._hashes
        assert "aaaaaaaa" not in logger._hashes

    def test_clear_hashes(self):
        """Test clearing all hashes"""
        logger = DashboardLogger(k=3, display=False)

        logger.update_hash("test_hash" + "0" * 56, HashStatus.SENT)
        assert len(logger._hashes) == 1

        logger.clear()
        assert len(logger._hashes) == 0

    @patch("sys.stdout.write")
    @patch("shutil.get_terminal_size")
    def test_console_update_disabled(self, mock_terminal_size, mock_stdout):
        """Test that console update is disabled when display=False"""
        mock_terminal_size.return_value.columns = 80

        logger = DashboardLogger(k=3, display=False)
        logger.update_hash("test_hash" + "0" * 56, HashStatus.SENT)

        # stdout.write should not be called when display is disabled
        mock_stdout.assert_not_called()

    @patch("sys.stdout.write")
    @patch("shutil.get_terminal_size")
    def test_console_update_enabled(self, mock_terminal_size, mock_stdout):
        """Test that console update works when display=True"""
        mock_terminal_size.return_value.columns = 80

        logger = DashboardLogger(k=3, display=True)
        logger.update_hash("test_hash" + "0" * 56, HashStatus.SENT)

        # stdout.write should be called when display is enabled
        mock_stdout.assert_called()


class TestCallMatching:
    """Test call identifier matching logic"""

    def test_call_matches_identical(self):
        """Test that identical calls match"""
        call1 = self._create_call_id("agent1", "hash123", 1, 100)
        call2 = self._create_call_id("agent1", "hash123", 1, 100)

        assert _call_matches(call1, call2) is True

    def test_call_matches_different_session(self):
        """Test that calls with different session_id still match"""
        call1 = self._create_call_id("agent1", "hash123", 1, 100)
        call2 = self._create_call_id("agent1", "hash123", 1, 200)  # Different session

        # Should match because session_id is ignored for matching
        assert _call_matches(call1, call2) is True

    def test_call_matches_different_hash(self):
        """Test that calls with different hashes don't match"""
        call1 = self._create_call_id("agent1", "hash123", 1, 100)
        call2 = self._create_call_id("agent1", "hash456", 1, 100)  # Different hash

        assert _call_matches(call1, call2) is False

    def test_call_matches_different_seq_id(self):
        """Test that calls with different seq_id don't match"""
        call1 = self._create_call_id("agent1", "hash123", 1, 100)
        call2 = self._create_call_id("agent1", "hash123", 2, 100)  # Different seq_id

        assert _call_matches(call1, call2) is False

    def _create_call_id(
        self, agent_name, doc_hash, seq_id, session_id
    ) -> CallIdentifier:
        """Helper to create call identifiers"""
        return {
            "agent_name": agent_name,
            "doc_hash": doc_hash,
            "seq_id": seq_id,
            "session_id": session_id,
            "meta": {
                "provider_type": "openai",
                "tag": None,
            },
        }


class TestHashRegressions:
    """Regression tests that verify hash values remain stable."""

    def test_hash_golden_text_only(self):
        """Regression: Text-only hashing must remain stable"""
        result = compute_hash("Instructions", ["Document 1", "Document 2"])
        expected = "f7649a6338429d417aad789788bfa2c268153ba554a88a6e87516d224f5f0bb3"
        assert result == expected

    def test_hash_golden_empty_documents(self):
        """Regression: Instructions-only hashing must remain stable"""
        result = compute_hash("Just instructions", [])
        expected = "0c20f4a03fe4dcf22b2c17499f3ac135bdfebd281461e7247707b7dfdc3084ee"
        assert result == expected

    def test_hash_golden_with_salt(self):
        """Regression: Salted hashing must remain stable"""
        result = compute_hash("Instructions", ["Document"], salt="v1")
        expected = "55e5ba6f7cbc77bfe86c278b3913f5655f9302a0bda8c77114b32bbb5a64611e"
        assert result == expected

    def test_hash_golden_with_image(self):
        """Regression: Image hashing must remain stable"""
        img = Image.new("RGB", (5, 5), color="blue")
        result = compute_hash("Process images", ["text", img])
        expected = "641221c8a2a2deb07212389f8613595ad0b4090e0984defa0fbe78e4393d2264"
        assert result == expected

    def test_hash_golden_no_instructions(self):
        """Regression: Hashing without instructions must remain stable"""
        result = compute_hash(None, ["Document 1", "Document 2"])
        expected = "207b3643bcb25e27637ad23d18973777d0054bbe30893f8fb218eba5ab7ea19d"
        assert result == expected

    def test_hash_golden_single_document(self):
        """Regression: Single document hashing must remain stable"""
        result = compute_hash("Test", ["Single"])
        expected = "8b38b94f33b8890d45561e5514de8a7fafdf5c784f97b8384571e32e7e0963db"
        assert result == expected

    def test_hash_golden_order_1(self):
        """Regression: Order-dependent hashing (order 1) must remain stable"""
        result = compute_hash("Order test", ["doc1", "doc2"])
        expected = "f49a51c37c43f2cdf6f4e6537f1aa5963a0b35de2db104bafeba43bb2643f8b8"
        assert result == expected

    def test_hash_golden_order_2(self):
        """Regression: Order-dependent hashing (order 2) must remain stable"""
        result = compute_hash("Order test", ["doc2", "doc1"])
        expected = "a6274ac796a0b9512007c5875e563b7d5459b9d22c02d4012a1a88f8767f0c0b"
        assert result == expected

    def test_hash_golden_salt_v1(self):
        """Regression: Salted hashing with v1 salt must remain stable"""
        result = compute_hash("hello", ["content"], salt="v1")
        expected = "01247da9b530d7c8582f8a90939e2f6bb1e4519608baa1b5822804b466ed3ce2"
        assert result == expected

    def test_hash_golden_salt_v2(self):
        """Regression: Salted hashing with v2 salt must remain stable"""
        result = compute_hash("hello", ["content"], salt="v2")
        expected = "c32fc413996c89f3da6e74081b723fa90d08df7f525c14474ef79bdd402bc87b"
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
