"""
Unit tests for Gemini provider classes

Tests the Gemini provider functionality including:
- SyncGeminiProvider and AsyncGeminiProvider
- Document formatting for Gemini API
- Provider type handling
"""

from dotenv import load_dotenv
import pytest
from parallem.provider.google.adapter import _fix_docs_for_google
from parallem.types import LLMIdentity
from parallem.provider.google.sdk import (
    SyncGoogleProvider,
    GoogleProvider,
)


class TestGeminiDocumentFormatting:
    """Test document formatting for Gemini API"""

    def test_fix_docs_single_string(self):
        """Test fixing a single string document"""
        result = _fix_docs_for_google(["Hello world"])

        assert len(result) == 1
        assert result[0] == {"role": "user", "parts": [{"text": "Hello world"}]}

    def test_fix_docs_list_of_strings(self):
        """Test fixing a list of string documents"""
        docs = ["First message", "Second message"]
        result = _fix_docs_for_google(docs)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"role": "user", "parts": [{"text": "First message"}]}
        assert result[1] == {"role": "user", "parts": [{"text": "Second message"}]}


class TestGeminiProviders:
    """Test Gemini provider classes"""

    def test_provider_type(self):
        """Test that GeminiProvider has correct provider type"""
        assert GoogleProvider.provider_type == "google"

    @pytest.mark.skip("costs real money")
    def test_sync_gemini_provider_prepare_sync_call(self):
        """Test SyncGeminiProvider prepares sync callable correctly"""
        load_dotenv()
        from google import genai

        client = genai.Client()
        provider = SyncGoogleProvider(client=client)

        # Test prepare_sync_call returns a callable
        sync_callable = provider.prepare_sync_call(
            instructions="Please respond exactly with Hello World!",
            documents=["Test document"],
            llm=LLMIdentity("gemini-2.5-flash"),
        )

        # Verify it returns a callable
        assert callable(sync_callable)

        # Execute the callable
        result = sync_callable()
        parsed = provider.parse_response(result)

        # Verify result
        assert parsed.text == "Hello World!"
