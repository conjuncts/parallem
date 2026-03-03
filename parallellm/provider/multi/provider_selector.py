from typing import Literal

from parallellm.types import ProviderType


def dynamic_select_provider(
    provider_type: ProviderType,
    strategy: Literal["sync", "concurrent", "batch"],
    *,
    multi_allowed=False,
):
    if provider_type == "openai":
        from parallellm.provider.openai.sdk import (
            ConcurrentOpenAIProvider,
            SyncOpenAIProvider,
            BatchOpenAIProvider,
        )

        if strategy == "concurrent":
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            provider = ConcurrentOpenAIProvider(client=client)
        elif strategy == "batch":
            from openai import OpenAI

            client = OpenAI()
            provider = BatchOpenAIProvider(client=client)
        else:
            # For other strategies, default to sync for now
            from openai import OpenAI

            client = OpenAI()
            provider = SyncOpenAIProvider(client=client)
    elif provider_type == "google":
        from parallellm.provider.google.sdk import (
            ConcurrentGoogleProvider,
            BatchGoogleProvider,
            SyncGoogleProvider,
        )
        from google import genai

        if strategy == "concurrent":
            client = genai.Client()
            provider = ConcurrentGoogleProvider(client=client)
        elif strategy == "batch":
            client = genai.Client()
            provider = BatchGoogleProvider(client=client)
        else:
            client = genai.Client()
            provider = SyncGoogleProvider(client=client)
    elif provider_type == "anthropic":
        from parallellm.provider.anthropic.sdk import (
            ConcurrentAnthropicProvider,
            SyncAnthropicProvider,
        )

        if strategy == "concurrent":
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
            provider = ConcurrentAnthropicProvider(client=client)
        else:
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
