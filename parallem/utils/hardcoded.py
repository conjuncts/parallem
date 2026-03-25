from typing import Optional


def guess_provider_and_name(identity: str) -> tuple[Optional[str], str]:
    """
    Guess the provider type from the identity string and return the provider and model name.

    :param identity: The identity string to guess the provider from.
    :returns: A tuple containing the guessed provider type (ie. openai) or None if not identifiable,
              and the model name.
    """
    if identity is None:
        return None, None

    if "/" in identity:
        # split by first '/', assume format 'provider/model_name'
        provider, model_name = identity.split("/", 1)
        return provider, model_name

    # import openai.types.shared.chat_model
    _openai_prefixes = ["gpt-", "o1-", "o3-", "o4-", "chatgpt"]
    if any(identity.startswith(prefix) for prefix in _openai_prefixes):
        return "openai", identity
    elif identity in ["o1", "o3", "o4"]:
        return "openai", identity

    # import anthropic.types.model_param
    _anthropic_prefixes = ["claude-"]
    if any(identity.startswith(prefix) for prefix in _anthropic_prefixes):
        return "anthropic", identity

    # import google.genai._local_tokenizer_loader
    # Source of truth:
    # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
    # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions
    _google_prefixes = ["gemini-"]
    if any(identity.startswith(prefix) for prefix in _google_prefixes):
        return "google", identity

    return None, identity


def guess_provider(identity: str) -> Optional[str]:
    """
    Guess the provider type from the identity string.

    :param identity: The identity string to guess the provider from.
    :returns: The guessed provider type (ie. openai) or None if not identifiable.
    """
    return guess_provider_and_name(identity)[0]
