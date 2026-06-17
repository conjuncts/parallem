"""
Unit tests for LLMIdentity and provider guessing functionality

Tests the LLM identity system including:
- LLMIdentity creation and provider detection
- Provider guessing from identity strings
- String conversion for different providers
- Default provider behavior
"""

import pytest

from parallem.types import LLMIdentity
from parallem.utils.hardcoded import guess_provider_and_name


def test_identity_with_guessed_provider():
    """Test creating identity with provider guessing"""
    identity = LLMIdentity("gpt-4")
    assert identity.identity == "gpt-4"
    assert identity.model_name == "gpt-4"
    assert identity.provider_type == "openai"

    identity = LLMIdentity("chatgpt-3.5-turbo")
    assert identity.identity == "chatgpt-3.5-turbo"
    assert identity.model_name == "chatgpt-3.5-turbo"
    assert identity.provider_type == "openai"

    identity = LLMIdentity("claude-sonnet-3.5")
    assert identity.identity == "claude-sonnet-3.5"
    assert identity.model_name == "claude-sonnet-3.5"
    assert identity.provider_type == "anthropic"

    identity = LLMIdentity("gemini-2.5-flash")
    assert identity.identity == "gemini-2.5-flash"
    assert identity.model_name == "gemini-2.5-flash"
    assert identity.provider_type == "google"


def test_identity_unknown_provider():
    """Test creating identity with unknown provider"""
    with pytest.raises(ValueError):
        _ = LLMIdentity("unknown-model")


def test_to_str_with_explicit_provider():
    """Test string conversion with explicit provider"""
    identity = LLMIdentity("claude-sonnet-3.5", provider_type="openai")
    assert identity.model_name == "claude-sonnet-3.5"
    assert identity.provider_type == "openai"


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


def test_identity_hashable():
    """Test that LLMIdentity can be used as a dictionary key"""
    identity1 = LLMIdentity("gpt-4")
    identity2 = LLMIdentity("claude-sonnet-3.5")

    # Should be able to use as dict keys
    my_dict = {identity1: "openai_model", identity2: "anthropic_model"}

    assert my_dict[identity1] == "openai_model"
    assert my_dict[identity2] == "anthropic_model"


def test_identity_equality():
    """Test that LLMIdentity instances with same provider/model are equal"""
    identity1 = LLMIdentity("gpt-4")
    identity2 = LLMIdentity("gpt-4")

    assert identity1 == identity2
    assert hash(identity1) == hash(identity2)

    # Different identities should not be equal
    identity3 = LLMIdentity("gpt-3.5-turbo")
    assert identity1 != identity3
    assert hash(identity1) != hash(identity3)


def test_identity_set_operations():
    """Test that LLMIdentity works correctly in sets"""
    identity1 = LLMIdentity("gpt-4")
    identity2 = LLMIdentity("gpt-4")
    identity3 = LLMIdentity("claude-sonnet-3.5")

    identity_set = {identity1, identity2, identity3}

    # identity1 and identity2 are equal, so set should have 2 items
    assert len(identity_set) == 2
    assert identity1 in identity_set
    assert identity3 in identity_set
