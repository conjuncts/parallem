from unittest.mock import Mock

from pipelinellm.provider.anthropic.sdk import BatchAnthropicProvider
from pipelinellm.provider.multi.provider_selector import dynamic_select_provider


def test_dynamic_select_provider_anthropic_batch():
    client = Mock()

    provider = dynamic_select_provider(
        "anthropic",
        "batch",
        client=client,
    )

    assert isinstance(provider, BatchAnthropicProvider)
    assert provider.client is client
