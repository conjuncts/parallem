import pytest
from pydantic import BaseModel

from parallem.core.exception import ProviderCompatibilityError
from parallem.provider.anthropic import _version_checks
from parallem.provider.anthropic.sdk import (
    AnthropicProvider,
)
from parallem.provider.anthropic.adapter import AnthropicAdapter
from parallem.types import LLMIdentity
from parallem.utils._quick_structured import _anthropic_transform_schema


class MyModel(BaseModel):
    final_answer: str


def _params(*, structured_output=None):
    return {
        "instructions": None,
        "strict_documents": ["What is the capital of France?"],
        "llm": LLMIdentity("claude-3-haiku-20240307", provider_type="anthropic"),
        "structured_output": structured_output,
        "tools": None,
    }


def _prepare_anthropic_config(params, **kwargs):
    return AnthropicAdapter().prepare_sdk_request(params, **kwargs)


def test_prepare_anthropic_config_with_pydantic_structured_output(monkeypatch):
    from parallem.provider.anthropic import adapter as anthropic_adapter

    monkeypatch.setattr(
        anthropic_adapter,
        "_transform_schema",
        _anthropic_transform_schema,
        raising=True,
    )
    model_name, messages, config = _prepare_anthropic_config(_params(structured_output=MyModel))

    assert model_name == "claude-3-haiku-20240307"
    assert messages == [{"role": "user", "content": "What is the capital of France?"}]
    assert config["output_config"]["format"]["type"] == "json_schema"
    assert config["output_config"]["format"]["schema"]["title"] == "MyModel"
    assert config["output_config"]["format"]["schema"]["additionalProperties"] is False


def test_prepare_anthropic_config_with_json_schema_dict(monkeypatch):
    from parallem.provider.anthropic import adapter as anthropic_adapter

    monkeypatch.setattr(
        anthropic_adapter,
        "_transform_schema",
        _anthropic_transform_schema,
        raising=True,
    )
    schema = {
        "type": "object",
        "properties": {"capital": {"type": "string"}},
        "required": ["capital"],
    }

    _, _, config = _prepare_anthropic_config(_params(structured_output=schema))

    format_payload = config["output_config"]["format"]
    assert format_payload["type"] == "json_schema"
    assert format_payload["schema"]["type"] == "object"
    assert format_payload["schema"]["additionalProperties"] is False
    assert format_payload["schema"]["properties"] == schema["properties"]


def test_prepare_anthropic_config_rejects_conflicting_output_format():
    with pytest.raises(AssertionError, match="Cannot supply both structured_output"):
        _prepare_anthropic_config(
            _params(structured_output=MyModel),
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {"type": "object", "properties": {}},
                }
            },
        )


def test_prepare_anthropic_config_rejects_invalid_structured_output():
    with pytest.raises(ValueError, match="Unsupported structured_output for Anthropic"):
        _prepare_anthropic_config(_params(structured_output="not-a-schema"))


def test_validate_request_compatibility_enforces_min_version(monkeypatch):
    monkeypatch.setattr(
        _version_checks.importlib_metadata,
        "version",
        lambda package_name: "0.76.9",
    )
    provider = AnthropicProvider()

    with pytest.raises(ProviderCompatibilityError, match="requires anthropic>=0.77.0"):
        provider.validate_request_compatibility(_params(structured_output=MyModel))


def test_validate_request_compatibility_allows_minimum_supported_version(monkeypatch):
    monkeypatch.setattr(
        _version_checks.importlib_metadata,
        "version",
        lambda package_name: "0.77.0",
    )
    provider = AnthropicProvider()

    provider.validate_request_compatibility(_params(structured_output=MyModel))


def test_validate_request_compatibility_skips_version_check_without_structured_output(
    monkeypatch,
):
    def _fail_if_called(package_name):
        raise AssertionError("version() should not be called")

    monkeypatch.setattr(
        _version_checks.importlib_metadata,
        "version",
        _fail_if_called,
    )

    provider = AnthropicProvider()
    provider.validate_request_compatibility(_params(structured_output=None))
