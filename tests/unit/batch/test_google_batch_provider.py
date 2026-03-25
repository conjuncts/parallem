from unittest.mock import Mock

from parallem.provider.google.sdk import BatchGoogleProvider


def test_cancel_batch_calls_google_batches_cancel_with_prefixed_name():
    mock_client = Mock()
    provider = BatchGoogleProvider(client=mock_client)

    provider.cancel_batch("batch_123", provider_type="google")

    mock_client.batches.cancel.assert_called_once_with(name="batches/batch_123")
