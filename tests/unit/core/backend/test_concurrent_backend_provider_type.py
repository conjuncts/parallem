import tempfile
from pathlib import Path
from typing import Optional

from parallem.core.backend.async_backend import AsyncBackend
from parallem.core.file_manager import FileManager
from parallem.provider.base import AsyncProvider
from parallem.types import LLMIdentity, ParsedResponse


class StubAsyncProvider(AsyncProvider):
    provider_type = "multi"

    def __init__(self):
        self.seen_provider_types = []

    def get_default_llm_identity(self):
        return LLMIdentity("gpt-5-nano", provider_type="openai")

    def validate_request_compatibility(self, params, **kwargs):
        return None

    def prepare_async_call(self, params, **kwargs):
        async def _coro():
            return {"content": "ok"}

        return _coro()

    def parse_response(self, raw_response, llm: Optional[LLMIdentity] = None):
        provider_type = llm.provider_type if llm else "unknown"
        self.seen_provider_types.append(provider_type)
        return ParsedResponse(
            text=f"{provider_type}:{raw_response['content']}",
            response_id=None,
            metadata=None,
        )


def test_async_backend_passes_llm_provider_type_to_parse_response():
    with tempfile.TemporaryDirectory() as tmp:
        file_manager = FileManager(Path(tmp))
        backend = AsyncBackend(file_manager)
        provider = StubAsyncProvider()

        call_id = {
            "agent_name": "test_agent",
            "doc_hash": "hash-provider-type",
            "seq_id": 1,
            "session_id": 1,
            "meta": {"provider_type": "multi", "tag": None},
        }

        params = {
            "instructions": "test",
            "strict_documents": [],
            "llm": LLMIdentity("gpt-5-nano", provider_type="openai"),
            "structured_output": str,
            "tools": None,
        }

        backend.call_llm(provider, params, call_id=call_id)

        parsed = backend.retrieve(call_id)
        assert parsed is not None
        assert parsed.text == "openai:ok"
        assert provider.seen_provider_types == ["openai"]

        backend.shutdown()
