"""
Unit tests for LLMIdentity and provider guessing functionality

Tests the LLM identity system including:
- LLMIdentity creation and provider detection
- Provider guessing from identity strings
- String conversion for different providers
- Default provider behavior
"""

from parallellm.types import LLMIdentity
from parallellm.utils.hardcoded import guess_provider_and_name


def test_identity_with_guessed_provider():
    """Test creating identity with provider guessing"""
    identity = LLMIdentity("gpt-4")
    assert identity.identity == "gpt-4"
    assert identity.model_name == "gpt-4"
    assert identity.provider == "openai"

    identity = LLMIdentity("chatgpt-3.5-turbo")
    assert identity.identity == "chatgpt-3.5-turbo"
    assert identity.model_name == "chatgpt-3.5-turbo"
    assert identity.provider == "openai"

    identity = LLMIdentity("claude-sonnet-3.5")
    assert identity.identity == "claude-sonnet-3.5"
    assert identity.model_name == "claude-sonnet-3.5"
    assert identity.provider == "anthropic"

    identity = LLMIdentity("gemini-2.5-flash")
    assert identity.identity == "gemini-2.5-flash"
    assert identity.model_name == "gemini-2.5-flash"
    assert identity.provider == "google"


def test_identity_unknown_provider():
    """Test creating identity with unknown provider"""
    identity = LLMIdentity("unknown-model")

    assert identity.identity == "unknown-model"
    assert identity.provider is None


def test_to_str_with_explicit_provider():
    """Test string conversion with explicit provider"""
    identity = LLMIdentity("claude-sonnet-3.5", provider="openai")
    assert identity.model_name == "claude-sonnet-3.5"
    assert identity.provider == "openai"


def test_guess_openai_provider():
    """Test guessing OpenAI provider from gpt- prefix"""
    assert guess_provider_and_name("gpt-4") == ("openai", "gpt-4")
    assert guess_provider_and_name("gpt-3.5-turbo") == ("openai", "gpt-3.5-turbo")
    assert guess_provider_and_name("gpt-4o") == ("openai", "gpt-4o")
    assert guess_provider_and_name("gpt-4-turbo") == ("openai", "gpt-4-turbo")


def test_guess_other_providers():
    assert guess_provider_and_name("claude-3") == ("anthropic", "claude-3")
    assert guess_provider_and_name("llama-2") == (None, "llama-2")
    assert guess_provider_and_name("custom-model") == (None, "custom-model")
    assert guess_provider_and_name("gemini-pro") == ("google", "gemini-pro")


def test_guess_edges():
    assert guess_provider_and_name(None) == (None, None)
    assert guess_provider_and_name("") == (None, "")

    # case sensitivity
    assert guess_provider_and_name("GPT-4") == (None, "GPT-4")
    assert guess_provider_and_name("Gpt-4") == (None, "Gpt-4")


def test_split_notation():
    assert guess_provider_and_name("openai/claude-sonnet-3.5") == (
        "openai",
        "claude-sonnet-3.5",
    )
