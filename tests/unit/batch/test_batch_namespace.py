from unittest.mock import Mock
import tempfile
from pathlib import Path

import pytest

from parallem.core.batch_namespace import BatchNamespace
from parallem.core.compress.pack_zip import read_jsonl_items_from_zip
from parallem.core.file_manager import FileManager
from parallem.provider.base import BatchProvider


class DummyBatchProvider(BatchProvider):
    provider_type = "openai"

    def __init__(self):
        self.cancelled_batches = []

    def cancel_batch(self, batch_uuid: str, provider_type: str) -> None:
        self.cancelled_batches.append((batch_uuid, provider_type))


@pytest.fixture
def mock_backend_datastore():
    datastore = Mock()
    backend = Mock()
    backend._get_datastore.return_value = datastore
    return backend, datastore


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def file_manager(temp_dir):
    return FileManager(temp_dir)


def test_forget_batch_clears_local_pending_without_cancel(mock_backend_datastore):
    backend, datastore = mock_backend_datastore
    provider = DummyBatchProvider()
    orch = Mock()
    orch._backend = backend
    orch._provider = provider

    ns = BatchNamespace(orch)
    ns.forget_batch("batch_123", provider_type="openai")

    assert provider.cancelled_batches == []
    datastore.clear_batch_pending.assert_called_once_with("batch_123")


def test_forget_batch_with_cancel_calls_provider_then_clears(mock_backend_datastore):
    backend, datastore = mock_backend_datastore
    provider = DummyBatchProvider()
    orch = Mock()
    orch._backend = backend
    orch._provider = provider

    ns = BatchNamespace(orch)
    ns.forget_batch("batch_123", provider_type="openai", cancel=True)

    assert provider.cancelled_batches == [("batch_123", "openai")]
    datastore.clear_batch_pending.assert_called_once_with("batch_123")


def test_forget_batch_rejects_incompatible_provider(mock_backend_datastore):
    backend, datastore = mock_backend_datastore
    provider = DummyBatchProvider()
    orch = Mock()
    orch._backend = backend
    orch._provider = provider

    ns = BatchNamespace(orch)

    with pytest.raises(ValueError, match="not compatible"):
        ns.forget_batch("batch_123", provider_type="google", cancel=True)

    assert provider.cancelled_batches == []
    datastore.clear_batch_pending.assert_not_called()


def test_forget_batch_cancel_requires_batch_provider(mock_backend_datastore):
    backend, datastore = mock_backend_datastore
    provider = Mock()
    provider.provider_type = "openai"
    provider.is_compatible.return_value = True

    orch = Mock()
    orch._backend = backend
    orch._provider = provider

    ns = BatchNamespace(orch)

    with pytest.raises(TypeError, match="does not support batch cancellation"):
        ns.forget_batch("batch_123", provider_type="openai", cancel=True)

    datastore.clear_batch_pending.assert_not_called()


def test_compress_inputs_writes_companion_parquet(file_manager):
    orch = Mock()
    orch._fm = file_manager
    orch._backend = Mock()
    orch._provider = Mock()

    ns = BatchNamespace(orch)

    file_manager.save_batch_in(
        [
            {
                "custom_id": "req_1",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": "gpt-4o-mini",
                    "instructions": "Be concise",
                    "input": [{"role": "user", "content": "Hi"}],
                    "tools": [],
                },
            }
        ]
    )
    file_manager.save_batch_in(
        [
            {
                "custom_id": "req_2",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": "gpt-4o-mini",
                    "instructions": "Be concise",
                    "input": [{"role": "user", "content": "Hello"}],
                    "tools": [],
                },
            }
        ]
    )

    ns._compress_inputs(provider_type="openai")

    zip_files = sorted(file_manager.path_batch_in().glob("*.zip"))
    assert len(zip_files) == 2

    batch_custom_ids = []
    for zip_file in zip_files:
        items = read_jsonl_items_from_zip(zip_file)
        assert len(items) == 1
        batch_custom_ids.append(items[0]["custom_id"])

    assert sorted(batch_custom_ids) == ["req_1", "req_2"]


def test_compress_inputs_noop_for_unsupported_provider(file_manager):
    orch = Mock()
    orch._fm = file_manager
    orch._backend = Mock()
    orch._provider = Mock()

    ns = BatchNamespace(orch)

    file_manager.save_batch_in([{"arbitrary": "value"}])
    ns._compress_inputs(provider_type="google")

    assert not list(file_manager.path_batch_in().glob("*.zip"))
