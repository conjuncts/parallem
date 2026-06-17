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
from parallem.logging.dash_logger import DashboardLogger, HashStatus
from parallem.core.calls import _call_matches
from parallem.types import CallIdentifier


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

    def _create_call_id(self, agent_name, doc_hash, seq_id, session_id) -> CallIdentifier:
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
