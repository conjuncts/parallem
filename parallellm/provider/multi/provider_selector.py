from typing import Literal

from parallellm.types import ProviderType


def dynamic_select_provider(
    provider_type: ProviderType,
    strategy: Literal["sync", "concurrent", "batch"],
    *,
    multi_allowed=False,
    client=None,
):
    """
    Produces the correct parallellm provider object.
    """
    if provider_type == "openai":
        from parallellm.provider.openai.sdk import (
            ConcurrentOpenAIProvider,
            SyncOpenAIProvider,
            BatchOpenAIProvider,
        )

        if strategy == "concurrent":
            if client is None:
                from openai import AsyncOpenAI

                client = AsyncOpenAI()
            provider = ConcurrentOpenAIProvider(client=client)
        elif strategy == "batch":
            if client is None:
                from openai import OpenAI

                client = OpenAI()
            provider = BatchOpenAIProvider(client=client)
        else:
            # For other strategies, default to sync for now
            if client is None:
                from openai import OpenAI

                client = OpenAI()
            provider = SyncOpenAIProvider(client=client)
    elif provider_type == "google":
        from parallellm.provider.google.sdk import (
            ConcurrentGoogleProvider,
            BatchGoogleProvider,
            SyncGoogleProvider,
        )

        if client is None:
            from google import genai

            client = genai.Client()

        if strategy == "concurrent":
            provider = ConcurrentGoogleProvider(client=client)
        elif strategy == "batch":
            provider = BatchGoogleProvider(client=client)
        else:
            provider = SyncGoogleProvider(client=client)
    elif provider_type == "anthropic":
        from parallellm.provider.anthropic.sdk import (
            ConcurrentAnthropicProvider,
            SyncAnthropicProvider,
        )

        if strategy == "concurrent":
            if client is None:
                from anthropic import AsyncAnthropic

                client = AsyncAnthropic()
            provider = ConcurrentAnthropicProvider(client=client)
        else:
            if client is None:
                from anthropic import Anthropic

                client = Anthropic()
            provider = SyncAnthropicProvider(client=client)
    elif multi_allowed and provider_type == "multi":
        from parallellm.provider.multi.multiplexer import (
            SyncMultiProvider,
            ConcurrentMultiProvider,
            BatchMultiProvider,
        )

        if strategy == "concurrent":
            provider = ConcurrentMultiProvider()
        elif strategy == "batch":
            provider = BatchMultiProvider()
        else:
            provider = SyncMultiProvider()
    else:
        raise NotImplementedError(f"Provider '{provider_type}' not implemented yet")
    return provider
