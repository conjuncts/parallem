"""
Unit tests for BatchBackend.execute_batch

Tests the batch grouping functionality:
- Groups by LLMIdentity when partition_by_model_name=True
- Chunks groups into max_batch_size or less
- Properly organizes calls with custom IDs
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock
import pytest

from pipelinellm.core.backend.batch_backend import BatchBackend
from pipelinellm.core.file_manager import FileManager
from pipelinellm.logging.dash_logger import PrimitiveDashboardLogger
from pipelinellm.types import LLMIdentity, CallIdentifier


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def file_manager(temp_dir):
    """Create a FileManager instance for testing"""
    return FileManager(temp_dir)


@pytest.fixture
def mock_datastore():
    """Create a mock datastore that does nothing"""
    mock_ds = Mock()
    mock_ds.is_call_in_pending_batch.return_value = False
    mock_ds.store_pending_batch.return_value = None
    return mock_ds


@pytest.fixture
def mock_provider():
    """Create a mock provider that does nothing"""
    mock_prov = Mock()

    # Mock prepare_batch_call to return a simple dict
    mock_prov.prepare_batch_call.return_value = {"test": "data"}

    # Mock get_batch_custom_ids to return custom IDs based on data length
    def get_custom_ids(data, provider_type=None):
        return [f"custom_{i}" for i in range(len(data))]

    mock_prov.get_batch_custom_ids.side_effect = get_custom_ids

    # Mock submit_batch_to_provider to return a UUID
    mock_prov.submit_batch_to_provider.return_value = "batch_uuid_123"

    return mock_prov


@pytest.fixture
def batch_backend(file_manager, mock_datastore):
    """Create a BatchBackend with mocked datastore"""
    backend = BatchBackend(
        fm=file_manager,
        dashlog=PrimitiveDashboardLogger(),
        session_id=1,
        confirm_batch_submission=False,
    )
    # Replace the datastore with our mock
    backend._ds = mock_datastore
    return backend


def create_call_id(agent_name: str, seq_id: int) -> CallIdentifier:
    """Helper to create call identifiers"""
    return {
        "agent_name": agent_name,
        "doc_hash": f"hash_{agent_name}_{seq_id}",
        "seq_id": seq_id,
        "session_id": 1,
        "meta": {"provider_type": "openai", "tag": None},
    }


class TestBatchBackendExecuteBatch:
    """Test BatchBackend.execute_batch grouping functionality"""

    def test_groups_by_llm_identity(self, batch_backend, mock_provider):
        """Test that calls are grouped by LLMIdentity"""
        # Create different LLM identities
        llm1 = LLMIdentity(
            "gpt-4o-mini", provider_type="openai", model_name="gpt-4o-mini"
        )
        llm2 = LLMIdentity("gpt-4o", provider_type="openai", model_name="gpt-4o")

        # Add calls with different LLMs
        for i in range(4):
            call_id = create_call_id("agent1", i)
            llm = llm1 if i < 2 else llm2
            batch_backend.bookkeep_call(call_id, llm, {"data": f"call_{i}"})

        # Execute batch with partition_by_model_name=True
        cohort = batch_backend.execute_batch(
            mock_provider,
            PrimitiveDashboardLogger(),
            max_batch_size=10,
            partition_by_model_name=True,
        )

        # Should create 2 batches (one for each LLM)
        assert len(cohort.batch_ids) == 2

        # Verify calls to submit_batch_to_provider
        assert mock_provider.submit_batch_to_provider.call_count == 2

        # Check that each submission was with the correct LLM
        submitted_llms = [
            call[0][1] for call in mock_provider.submit_batch_to_provider.call_args_list
        ]
        assert llm1 in submitted_llms
        assert llm2 in submitted_llms

        # Buffer should be cleared
        assert len(batch_backend._batch_buffer) == 0

    def test_chunks_into_max_batch_size(self, batch_backend, mock_provider):
        """Test that large groups are chunked into max_batch_size"""
        llm = LLMIdentity(
            "gpt-4o-mini", provider_type="openai", model_name="gpt-4o-mini"
        )

        # Add 10 calls with the same LLM
        for i in range(10):
            call_id = create_call_id("agent1", i)
            batch_backend.bookkeep_call(call_id, llm, {"data": f"call_{i}"})

        # Execute batch with max_batch_size=3
        cohort = batch_backend.execute_batch(
            mock_provider,
            PrimitiveDashboardLogger(),
            max_batch_size=3,
            partition_by_model_name=True,
        )

        # Should create 4 batches (3+3+3+1)
        assert len(cohort.batch_ids) == 4

        # Verify each batch has at most 3 calls
        for batch_id in cohort.batch_ids:
            assert len(batch_id.call_ids) <= 3

        # Verify total calls remains 10
        total_calls = sum(len(batch_id.call_ids) for batch_id in cohort.batch_ids)
        assert total_calls == 10

    def test_uses_configured_default_max_batch_size(
        self, file_manager, mock_datastore, mock_provider
    ):
        """Test execute_batch uses backend-configured max_batch_size when omitted"""
        backend = BatchBackend(
            fm=file_manager,
            dashlog=PrimitiveDashboardLogger(),
            session_id=1,
            confirm_batch_submission=False,
            max_batch_size=3,
        )
        backend._ds = mock_datastore

        llm = LLMIdentity(
            "gpt-4o-mini", provider_type="openai", model_name="gpt-4o-mini"
        )

        for i in range(8):
            call_id = create_call_id("agent1", i)
            backend.bookkeep_call(call_id, llm, {"data": f"call_{i}"})

        cohort = backend.execute_batch(
            mock_provider,
            PrimitiveDashboardLogger(),
            partition_by_model_name=True,
        )

        assert len(cohort.batch_ids) == 3
        assert [len(batch_id.call_ids) for batch_id in cohort.batch_ids] == [3, 3, 2]

    def test_combined_grouping_and_chunking(self, batch_backend, mock_provider):
        """Test grouping by LLM and chunking together"""
        llm1 = LLMIdentity(
            "gpt-4o-mini", provider_type="openai", model_name="gpt-4o-mini"
        )
        llm2 = LLMIdentity("gpt-4o", provider_type="openai", model_name="gpt-4o")
        llm3 = LLMIdentity(
            "claude-3-5-sonnet-20241022",
            provider_type="anthropic",
            model_name="claude-3-5-sonnet-20241022",
        )

        # Add 5 calls for llm1, 8 calls for llm2, 2 calls for llm3
        for i in range(5):
            call_id = create_call_id("agent1", i)
            batch_backend.bookkeep_call(call_id, llm1, {"data": f"llm1_call_{i}"})

        for i in range(8):
            call_id = create_call_id("agent2", i)
            batch_backend.bookkeep_call(call_id, llm2, {"data": f"llm2_call_{i}"})

        for i in range(2):
            call_id = create_call_id("agent3", i)
            batch_backend.bookkeep_call(call_id, llm3, {"data": f"llm3_call_{i}"})

        # Execute batch with max_batch_size=3
        cohort = batch_backend.execute_batch(
            mock_provider,
            PrimitiveDashboardLogger(),
            max_batch_size=3,
            partition_by_model_name=True,
        )

        # Should create:
        # - llm1: 5 calls -> 2 batches (3+2)
        # - llm2: 8 calls -> 3 batches (3+3+2)
        # - llm3: 2 calls -> 1 batch (2)
        # Total: 6 batches
        assert len(cohort.batch_ids) == 6

        # Verify each batch has at most 3 calls
        for batch_id in cohort.batch_ids:
            assert len(batch_id.call_ids) <= 3

        # Verify total calls
        total_calls = sum(len(batch_id.call_ids) for batch_id in cohort.batch_ids)
        assert total_calls == 15

    def test_no_partition_by_model_name(self, batch_backend, mock_provider):
        """Test with partition_by_model_name=False"""
        llm1 = LLMIdentity(
            "gpt-4o-mini", provider_type="openai", model_name="gpt-4o-mini"
        )
        llm2 = LLMIdentity("gpt-4o", provider_type="openai", model_name="gpt-4o")

        # Add calls with different LLMs
        for i in range(4):
            call_id = create_call_id("agent1", i)
            llm = llm1 if i < 2 else llm2
            batch_backend.bookkeep_call(call_id, llm, {"data": f"call_{i}"})

        # Execute batch with partition_by_model_name=False and max_batch_size=3
        cohort = batch_backend.execute_batch(
            mock_provider,
            PrimitiveDashboardLogger(),
            max_batch_size=3,
            partition_by_model_name=False,
        )

        # Should create 2 batches (3+1) regardless of LLM
        assert len(cohort.batch_ids) == 2

        # Verify sizes
        assert len(cohort.batch_ids[0].call_ids) == 3
        assert len(cohort.batch_ids[1].call_ids) == 1

    def test_empty_buffer(self, batch_backend, mock_provider):
        """Test execute_batch with empty buffer"""
        cohort = batch_backend.execute_batch(
            mock_provider,
            PrimitiveDashboardLogger(),
            max_batch_size=3,
            partition_by_model_name=True,
        )

        # Should create no batches
        assert len(cohort.batch_ids) == 0
        assert cohort.session_id == 1

        # Should not call provider
        assert mock_provider.submit_batch_to_provider.call_count == 0

    def test_pending_requests_counting(self, batch_backend, mock_provider):
        """Test that pending requests are counted"""
        from pipelinellm.core.exception import PendingNotAvailable

        llm = LLMIdentity(
            "gpt-4o-mini", provider_type="openai", model_name="gpt-4o-mini"
        )

        # Add some calls
        call_ids = [create_call_id("agent1", i) for i in range(3)]
        for call_id in call_ids:
            batch_backend.bookkeep_call(call_id, llm, {"data": "test"})

        # Execute batch to store them as pending
        batch_backend.execute_batch(
            mock_provider, PrimitiveDashboardLogger(), max_batch_size=10
        )

        # Mock datastore to return True for pending checks
        batch_backend._ds.is_call_in_pending_batch.return_value = True

        # Try to submit the same calls again (they're now pending)
        for call_id in call_ids:
            try:
                batch_backend.submit_query(
                    mock_provider,
                    {
                        "instructions": None,
                        "strict_documents": [],
                        "llm": llm,
                        "structured_output": None,
                        "tools": None,
                    },
                    call_id=call_id,
                )
            except PendingNotAvailable:
                pass  # Expected

        # Check that the counter was incremented
        assert batch_backend._pending_count == 3

    @pytest.mark.skip(
        reason="This test doesn't work because of the way that the provider is mocked"
    )
    def test_custom_ids_generation(self, batch_backend, mock_provider):
        """Test that custom IDs are properly generated for each call"""
        llm = LLMIdentity(
            "gpt-4o-mini", provider_type="openai", model_name="gpt-4o-mini"
        )

        # Add 5 calls
        for i in range(5):
            call_id = create_call_id("agent1", i)
            batch_backend.bookkeep_call(call_id, llm, {"data": f"call_{i}"})

        # Execute batch
        cohort = batch_backend.execute_batch(
            mock_provider,
            PrimitiveDashboardLogger(),
            max_batch_size=3,
            partition_by_model_name=True,
        )

        # Verify custom IDs were generated
        # This test doesn't work because of the way that the provider is mocked
        for batch_id in cohort.batch_ids:
            # Assert all unique within batch. Doesn't actually test anything,
            # since this is being mocked anyways, but maybe in the future.
            assert len(batch_id.custom_ids) == len(set(batch_id.custom_ids))
