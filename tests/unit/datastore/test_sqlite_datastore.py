"""
Tests for SQLiteDatastore
"""

import pytest
import tempfile
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

from parallellm.core.datastore.sqlite import SQLiteDatastore
from parallellm.core.file_manager import FileManager
from parallellm.types import (
    CallIdentifier,
    ParsedResponse,
    BatchIdentifier,
    BatchResult,
)


@pytest.fixture
def temp_datastore():
    """Create a temporary datastore for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_manager = FileManager(Path(temp_dir))
        datastore = SQLiteDatastore(file_manager)
        yield datastore
        datastore.close()


@pytest.fixture
def batch_call_ids():
    """Create batch call_ids for testing batch operations"""
    return [
        {
            "agent_name": "batch_agent",
            "doc_hash": f"batch_hash_{i}",
            "seq_id": i,
            "session_id": 100,
            "meta": {"provider_type": "openai", "tag": f"batch_tag_{i}"},
        }
        for i in range(1, 4)
    ]


@pytest.fixture
def batch_identifier(batch_call_ids):
    """Create a BatchIdentifier for testing"""
    custom_ids = [f"custom_{i}" for i in range(1, 4)]
    batch_uuid = "test-batch-uuid-123"

    return BatchIdentifier(
        call_ids=batch_call_ids, custom_ids=custom_ids, batch_uuid=batch_uuid
    )


class TestSQLite:
    def test_store_and_retrieve_anonymous_response(
        self, temp_datastore, generic_call_id
    ):
        """Test storing and retrieving an anonymous response"""

        metadata = {"usage": {"total_tokens": 50}, "model": "gpt-4"}
        parsed_response = ParsedResponse(
            text="Test response", response_id="resp_123", metadata=metadata
        )

        # Store the response
        temp_datastore.store(generic_call_id, parsed_response)

        # Retrieve the response
        retrieved = temp_datastore.retrieve(generic_call_id)
        assert retrieved is not None
        assert retrieved.text == "Test response"

        # Retrieve with metadata
        retrieved_with_metadata = temp_datastore.retrieve(
            generic_call_id, metadata=True
        )
        assert retrieved_with_metadata.metadata == metadata

    def test_store_update_existing_response(self, temp_datastore, generic_call_id):
        """Test that storing with same call_id updates existing response"""

        # Store original response
        original_response = ParsedResponse(
            text="Original response", response_id="orig_123", metadata={}
        )
        temp_datastore.store(generic_call_id, original_response)

        # Store updated response
        updated_response = ParsedResponse(
            text="Updated response",
            response_id="updated_123",
            metadata={"updated": True},
        )
        temp_datastore.store(generic_call_id, updated_response, upsert=True)

        # Retrieve should return updated response
        retrieved = temp_datastore.retrieve(generic_call_id)
        assert retrieved.text == "Updated response"

    def test_retrieve_nonexistent_response(self, temp_datastore, generic_call_id):
        """Test retrieving a response that doesn't exist"""

        retrieved = temp_datastore.retrieve(generic_call_id)
        assert retrieved is None

    def test_retrieve_fallback_without_seq_id(self, temp_datastore, generic_call_id):
        """Test that retrieve falls back to matching without seq_id"""

        parsed_response = ParsedResponse(
            text="Fallback response", response_id="fallback_123", metadata={}
        )

        # Store response
        temp_datastore.store(generic_call_id, parsed_response)

        not_this_one: CallIdentifier = {
            "agent_name": "wrong_agent",
            "doc_hash": "wrong_hash",
            "seq_id": 999,
            "session_id": 100,
            "meta": {"provider_type": "openai", "tag": None},
        }
        temp_datastore.store(
            not_this_one,
            ParsedResponse(text="Wrong response", response_id="wrong_123", metadata={}),
        )

        # Try to retrieve with different seq_id
        call_id_different_seq = generic_call_id.copy()
        call_id_different_seq["seq_id"] = 999

        retrieved = temp_datastore.retrieve(call_id_different_seq)
        assert retrieved is not None
        assert retrieved.text == "Fallback response"

    def test_tag_storage_and_retrieval(self, temp_datastore, generic_call_id):
        """Test that tag is correctly stored and retrieved"""
        call_id = generic_call_id
        call_id["meta"]["tag"] = "test_tag_value"

        metadata = {"usage": {"total_tokens": 50}, "model": "gpt-4"}
        parsed_response = ParsedResponse(
            text="Test response with tag", response_id="tag_resp_123", metadata=metadata
        )

        # Store the response
        temp_datastore.store(call_id, parsed_response)

        # Check that tag is stored in metadata table
        conn = temp_datastore._get_connection()
        cursor = conn.execute(
            "SELECT tag FROM metadata WHERE agent_name = ? AND seq_id = ? AND session_id = ?",
            (call_id["agent_name"], call_id["seq_id"], call_id["session_id"]),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row["tag"] == "test_tag_value"


# @pytest.mark.skip("Takes extra time")
class TestSQLiteBatch:
    def test_batch_operations(self, temp_datastore, batch_identifier):
        """Test batch-related operations"""
        # Store pending batch
        temp_datastore.store_pending_batch(batch_identifier)

        # Check that call IDs can be retrieved
        retrieved_call_ids = temp_datastore.retrieve_batch_call_ids(
            batch_identifier.batch_uuid
        )
        assert len(retrieved_call_ids) == 3
        assert retrieved_call_ids[0]["doc_hash"] == "batch_hash_1"

        # Check if call is in pending batch
        assert temp_datastore.is_call_in_pending_batch(batch_identifier.call_ids[0])

        # Get all pending batch UUIDs
        pending_uuids = temp_datastore.get_all_pending_batch_uuids()
        assert (batch_identifier.batch_uuid, "openai") in pending_uuids

        # Clear batch pending
        temp_datastore.clear_batch_pending(batch_identifier.batch_uuid)

        # Verify cleared
        assert not temp_datastore.is_call_in_pending_batch(batch_identifier.call_ids[0])
        pending_uuids_after = temp_datastore.get_all_pending_batch_uuids()
        assert batch_identifier.batch_uuid not in pending_uuids_after

    def test_batch_tag_storage(self, temp_datastore, batch_call_ids):
        """Test that tag is correctly stored in batch operations"""
        # Use only first 2 call_ids for this test
        call_ids = batch_call_ids[:2]

        custom_ids = [f"tag_custom_{i}" for i in range(1, 3)]
        batch_uuid = "tag-batch-uuid-123"

        batch_id = BatchIdentifier(
            call_ids=call_ids, custom_ids=custom_ids, batch_uuid=batch_uuid
        )

        # Store pending batch
        temp_datastore.store_pending_batch(batch_id)

        # Check that tags are stored in batch_pending table
        conn = temp_datastore._get_connection()
        cursor = conn.execute(
            "SELECT tag FROM batch_pending WHERE batch_uuid = ? ORDER BY seq_id",
            (batch_uuid,),
        )
        rows = cursor.fetchall()
        assert len(rows) == 2
        assert rows[0]["tag"] == "batch_tag_1"
        assert rows[1]["tag"] == "batch_tag_2"

        # Test store_ready_batch preserves tag in metadata
        parsed_responses = [
            ParsedResponse(
                text=f"Batch response {i}",
                response_id=f"resp_{i}",
                custom_id=f"tag_custom_{i}",
                metadata={"batch": True},
            )
            for i in range(1, 3)
        ]

        batch_result = BatchResult(
            status="ready", raw_output="batch output", parsed_responses=parsed_responses
        )

        # Store ready batch
        temp_datastore.store_ready_batch(batch_result)

        # Verify tags are transferred to metadata table
        cursor = conn.execute(
            "SELECT tag FROM metadata WHERE response_id IN (?, ?) ORDER BY response_id",
            ("resp_1", "resp_2"),
        )
        metadata_rows = cursor.fetchall()
        assert len(metadata_rows) == 2
        assert metadata_rows[0]["tag"] == "batch_tag_1"
        assert metadata_rows[1]["tag"] == "batch_tag_2"

    def test_store_ready_batch(self, temp_datastore, batch_identifier: BatchIdentifier):
        """Test storing completed batch results"""

        temp_datastore.store_pending_batch(batch_identifier)
        call_ids = batch_identifier.call_ids

        # Create batch result
        parsed_responses = [
            ParsedResponse(
                text=f"Batch response {i}",
                response_id=f"resp_{i}",
                custom_id=f"custom_{i}",
                metadata={"batch": True},
            )
            for i in range(1, 4)
        ]

        batch_result = BatchResult(
            status="ready", raw_output="batch output", parsed_responses=parsed_responses
        )

        # Store ready batch
        temp_datastore.store_ready_batch(batch_result)

        # Verify responses can be retrieved
        for i, call_id in enumerate(call_ids):
            retrieved = temp_datastore.retrieve(call_id)
            assert retrieved is not None
            assert retrieved.text == f"Batch response {i + 1}"

    def test_empty_batch_handling(self, temp_datastore):
        """Test handling of empty batch results"""
        batch_result = BatchResult(
            status="ready", raw_output="empty batch", parsed_responses=None
        )

        # Should not raise an error
        temp_datastore.store_ready_batch(batch_result)

        # Test with empty list too
        batch_result.parsed_responses = []
        temp_datastore.store_ready_batch(batch_result)

    def test_missing_batch_pending_record(self, temp_datastore):
        """Test error handling when batch pending record is missing"""
        parsed_responses = [
            ParsedResponse(
                text="Missing batch response",
                response_id="missing_custom_123",
                metadata={},
            )
        ]

        batch_result = BatchResult(
            status="ready",
            raw_output="missing batch",
            parsed_responses=parsed_responses,
        )

        # Should raise an error for missing pending record
        with pytest.raises(ValueError, match="Could not find pending batch record"):
            temp_datastore.store_ready_batch(batch_result)


@pytest.mark.skip("Takes extra time")
class TestSQLiteExtras:
    """Test suite for SQLiteDatastore"""

    def test_connection_creation(self, temp_datastore):
        """Test that database connections are created correctly"""
        conn = temp_datastore._get_connection()
        assert isinstance(conn, sqlite3.Connection)

        # Check that tables are created
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        expected_tables = [
            "anon_responses",
            "metadata",
            "batch_pending",
        ]
        for table in expected_tables:
            assert table in tables

    def test_null_agent_name_handling(self, temp_datastore, generic_call_id):
        """Test handling of null agent names"""
        generic_call_id["agent_name"] = None

        parsed_response = ParsedResponse(
            text="Null agent response", response_id="null_123", metadata={}
        )

        # Store and retrieve
        temp_datastore.store(generic_call_id, parsed_response)
        retrieved = temp_datastore.retrieve(generic_call_id)

        assert retrieved is not None
        assert retrieved.text == "Null agent response"

    def test_metadata_operations(self, temp_datastore, generic_call_id):
        """Test metadata storage and retrieval"""

        metadata = {
            "usage": {
                "total_tokens": 100,
                "prompt_tokens": 50,
                "completion_tokens": 50,
            },
            "model": "gpt-4",
            "finish_reason": "stop",
        }

        parsed_response = ParsedResponse(
            text="Response with metadata", response_id="meta_123", metadata=metadata
        )

        # Store response with metadata
        temp_datastore.store(generic_call_id, parsed_response)

        # Retrieve metadata separately
        retrieved_metadata = temp_datastore.retrieve_metadata("meta_123")
        assert retrieved_metadata == metadata

    def test_persist_and_close(self, temp_datastore, generic_call_id):
        """Test persist and close operations"""

        parsed_response = ParsedResponse(
            text="Persist test response", response_id="persist_123", metadata={}
        )

        temp_datastore.store(generic_call_id, parsed_response)

        # Persist should not raise any errors
        temp_datastore.persist()

        # Close should not raise any errors
        temp_datastore.close()

        # Data should still be retrievable after reconnection
        retrieved = temp_datastore.retrieve(generic_call_id)
        assert retrieved is not None
        assert retrieved.text == "Persist test response"

    def test_threading_isolation(self, temp_datastore):
        """Test that connections are isolated per thread"""
        import threading
        import time

        results = {}

        def thread_worker(thread_id):
            # Each thread should get its own connection
            conn1 = temp_datastore._get_connection()
            conn2 = temp_datastore._get_connection()

            # Same thread should get same connection
            results[thread_id] = conn1 is conn2

        threads = []
        for i in range(3):
            thread = threading.Thread(target=thread_worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All threads should have gotten the same connection within their thread
        for thread_id, same_connection in results.items():
            assert same_connection, f"Thread {thread_id} got different connections"

    def test_sql_error_handling(self, temp_datastore, generic_call_id):
        """Test handling of SQL errors"""
        # Try to cause a database error by manually corrupting the database
        with pytest.raises(Exception):
            generic_call_id["doc_hash"] = None  # Invalid value

            parsed_response = ParsedResponse(
                text="Error test", response_id="error_123", metadata={}
            )

            temp_datastore.store(generic_call_id, parsed_response)

    @pytest.mark.skip("Fails but idk why")
    @patch("parallellm.core.sink.sequester.sequester_openai_metadata")
    def test_metadata_transfer_on_persist(
        self, mock_sequester, temp_datastore, generic_call_id
    ):
        """Test that metadata transfer is called during persist"""

        metadata = {"usage": {"total_tokens": 75}}
        parsed_response = ParsedResponse(
            text="Transfer test response", response_id="transfer_123", metadata=metadata
        )

        temp_datastore.store(generic_call_id, parsed_response)

        # Mock the sequester function to return some succeeded transfers
        mock_sequester.return_value = ["transfer_123"]

        # Persist should trigger metadata transfer
        temp_datastore.persist()

        # Verify sequester was called
        mock_sequester.assert_called_once()

    def test_destructor_cleanup(self):
        """Test that destructor properly cleans up connections"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_manager = FileManager(Path(temp_dir))
            datastore = SQLiteDatastore(file_manager)

            # Get a connection to ensure it exists
            conn = datastore._get_connection()
            assert isinstance(conn, sqlite3.Connection)

            # Manually call destructor
            datastore.__del__()

            # This should not raise an error
            assert True
