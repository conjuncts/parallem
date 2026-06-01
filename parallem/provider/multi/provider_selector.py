import os
from typing import Literal

from parallem.types import ProviderType


def dynamic_select_provider(
    provider_type: ProviderType,
    strategy: Literal["sync", "async", "batch"],
    *,
    multi_allowed=False,
    client=None,
):
    """
    Produces the correct provider object.
    """
    if provider_type == "openai":
        from parallem.provider.openai.sdk import (
            AsyncOpenAIProvider,
            SyncOpenAIProvider,
            BatchOpenAIProvider,
        )

        if strategy == "async":
            if client is None:
                from openai import AsyncOpenAI

                client = AsyncOpenAI()
            provider = AsyncOpenAIProvider(client=client)
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
            AsyncOpenAIChatProvider,
            SyncOpenAIChatProvider,
        )

        if strategy == "async":
            if client is None:
                from openai import AsyncOpenAI

                client = AsyncOpenAI()
            provider = AsyncOpenAIChatProvider(client=client)
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
            AsyncGoogleProvider,
            BatchGoogleProvider,
            SyncGoogleProvider,
        )

        if client is None:
            from google import genai

            client = genai.Client()

        if strategy == "async":
            provider = AsyncGoogleProvider(client=client)
        elif strategy == "batch":
            provider = BatchGoogleProvider(client=client)
        else:
            provider = SyncGoogleProvider(client=client)
    elif provider_type == "anthropic":
        from parallem.provider.anthropic.sdk import (
            BatchAnthropicProvider,
            AsyncAnthropicProvider,
            SyncAnthropicProvider,
        )

        if strategy == "async":
            if client is None:
                from anthropic import AsyncAnthropic

                client = AsyncAnthropic()
            provider = AsyncAnthropicProvider(client=client)
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
            AsyncBedrockProvider,
            SyncBedrockProvider,
        )
        from parallem.provider.bedrock.sdk_s3 import BatchBedrockProvider

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

        if strategy == "async":
            provider = AsyncBedrockProvider(client=client)
        elif strategy == "batch":
            provider = BatchBedrockProvider(client=client)
        else:
            provider = SyncBedrockProvider(client=client)
    elif multi_allowed and provider_type == "multi":
        from parallem.provider.multi.multiplexer import (
            SyncMultiProvider,
            AsyncMultiProvider,
            BatchMultiProvider,
        )

        if strategy == "async":
            provider = AsyncMultiProvider()
        elif strategy == "batch":
            provider = BatchMultiProvider()
        else:
            provider = SyncMultiProvider()
    else:
        raise NotImplementedError(f"Provider '{provider_type}' not implemented yet")
    return provider
