from unittest.mock import Mock

import pytest

from parallem.core.batch_namespace import BatchNamespace
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
