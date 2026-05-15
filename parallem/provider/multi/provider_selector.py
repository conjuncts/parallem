import os
from typing import Literal

from parallem.types import ProviderType


def dynamic_select_provider(
    provider_type: ProviderType,
    strategy: Literal["sync", "concurrent", "batch"],
    *,
    multi_allowed=False,
    client=None,
):
    """
    Produces the correct provider object.
    """
    if provider_type == "openai":
        from parallem.provider.openai.sdk import (
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
    elif provider_type == "openai-chat":
        from parallem.provider.openai_chat.sdk import (
            BatchOpenAIChatProvider,
            ConcurrentOpenAIChatProvider,
            SyncOpenAIChatProvider,
        )

        if strategy == "concurrent":
            if client is None:
                from openai import AsyncOpenAI

                client = AsyncOpenAI()
            provider = ConcurrentOpenAIChatProvider(client=client)
        elif strategy == "batch":
            if client is None:
                from openai import OpenAI

                client = OpenAI()
            provider = BatchOpenAIChatProvider(client=client)
        else:
            if client is None:
                from openai import OpenAI

                client = OpenAI()
            provider = SyncOpenAIChatProvider(client=client)
    elif provider_type == "google":
        from parallem.provider.google.sdk import (
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
        from parallem.provider.anthropic.sdk import (
            BatchAnthropicProvider,
            ConcurrentAnthropicProvider,
            SyncAnthropicProvider,
        )

        if strategy == "concurrent":
            if client is None:
                from anthropic import AsyncAnthropic

                client = AsyncAnthropic()
            provider = ConcurrentAnthropicProvider(client=client)
        elif strategy == "batch":
            if client is None:
                from anthropic import Anthropic

                client = Anthropic()
            provider = BatchAnthropicProvider(client=client)
        else:
            if client is None:
                from anthropic import Anthropic

                client = Anthropic()
            provider = SyncAnthropicProvider(client=client)
    elif provider_type == "bedrock":
        from parallem.provider.bedrock.sdk import (
            ConcurrentBedrockProvider,
            SyncBedrockProvider,
        )

        if client is None:
            # boto3 needed for Amazon Bedrock - pip install boto3
            import boto3

            region_name = os.getenv("AWS_REGION")
            if region_name is None:
                raise ValueError(
                    "Please set the AWS_REGION to use Amazon Bedrock. "
                    "Example: export AWS_REGION=us-east-1"
                )
            client = boto3.client("bedrock-runtime", region_name=region_name)

        if strategy == "concurrent":
            provider = ConcurrentBedrockProvider(client=client)
        elif strategy == "batch":
            raise NotImplementedError("Bedrock provider does not support batch mode.")
        else:
            provider = SyncBedrockProvider(client=client)
    elif multi_allowed and provider_type == "multi":
        from parallem.provider.multi.multiplexer import (
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
        from parallem.registry.provider_registry import get_provider, is_registered

        if is_registered(provider_type):
            provider = get_provider(provider_type, strategy, client=client)
        else:
            raise NotImplementedError(f"Provider '{provider_type}' not implemented yet")
    return provider
