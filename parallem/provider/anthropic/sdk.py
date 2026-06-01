import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
from pydantic import BaseModel
from parallem.provider.anthropic.adapter import AnthropicAdapter
from parallem.provider.anthropic._version_checks import (
    enforce_anthropic_min_version_for_structured_output,
)
from parallem.provider.base import (
    BatchProvider,
    AsyncProvider,
    BaseProvider,
    SyncProvider,
)
from parallem.types import (
    BatchResult,
    ParsedResponse,
    ParsedError,
    CommonQueryParameters,
    LLMIdentity,
)
from parallem.utils._batch_helper import _split_batch_response

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic


class AnthropicProvider(BaseProvider):
    provider_type = "anthropic"

    def __init__(self):
        self.adapter = AnthropicAdapter()

    def validate_request_compatibility(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> None:
        """Validate Anthropic request compatibility.

        :param params: Common query parameters for the request.
        :return: None.
        """
        structured_output = params.get("structured_output")
        if structured_output is not None:
            enforce_anthropic_min_version_for_structured_output()

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("claude-haiku-4-5-20251001", provider_type=self.provider_type)

    def parse_response(
        self, raw_response: Union[BaseModel, dict], llm: Optional[LLMIdentity] = None
    ) -> ParsedResponse:
        """Parse Anthropic API response into common format"""

        return self.adapter.convert_response(raw_response)


class SyncAnthropicProvider(SyncProvider, AnthropicProvider):
    def __init__(self, client: "Anthropic"):
        super().__init__()
        self.client = client

    def prepare_sync_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a synchronous callable for Anthropic API"""
        model_name, messages, config = self.adapter.fix_config(params, **kwargs)
        max_tokens = config.pop("max_tokens", 4096)
        # If betas are requested and the client exposes the beta namespace,
        # use the beta messages.create endpoint.
        if config.get("betas"):
            return self.client.beta.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=messages,
                **config,
            )

        return self.client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=messages,
            **config,
        )


class AsyncAnthropicProvider(AsyncProvider, AnthropicProvider):
    def __init__(self, client: "AsyncAnthropic"):
        super().__init__()
        self.client = client

    def prepare_async_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a async coroutine for Anthropic API"""
        model_name, messages, config = self.adapter.fix_config(params, **kwargs)
        max_tokens = config.pop("max_tokens", 1024)

        if config.get("betas"):
            coro = self.client.beta.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=messages,
                **config,
            )
        else:
            coro = self.client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=messages,
                **config,
            )

        return coro


class BatchAnthropicProvider(BatchProvider, AnthropicProvider):
    def __init__(self, client: "Anthropic"):
        super().__init__()
        self.client = client

    def prepare_batch_call(
        self,
        params: CommonQueryParameters,
        custom_id: str,
        **kwargs,
    ) -> dict:
        """Convert CommonQueryParameters to Anthropic Message Batch format."""
        request_params = self.adapter.prepare_request(params, **kwargs)

        return {
            "custom_id": custom_id,
            "params": request_params,
        }

    def get_batch_custom_ids(self, stuff: list[dict], provider_type: str) -> list[str]:
        custom_ids = []
        for item in stuff:
            custom_id = item.get("custom_id")
            if not custom_id:
                raise ValueError("Each batch item must have a 'custom_id' field.")
            custom_ids.append(custom_id)
        return custom_ids

    def submit_batch_to_provider(self, fpath: Path, llm: LLMIdentity) -> str:
        """Submit JSONL requests as an Anthropic Message Batch and return batch ID."""
        requests = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                requests.append(json.loads(line))

        batch = self.client.messages.batches.create(requests=requests)
        return batch.id

    def cancel_batch(self, batch_uuid: str, provider_type: str) -> None:
        """Cancel an Anthropic Message Batch."""
        self.client.messages.batches.cancel(batch_uuid)

    def _decode_anthropic_batch_success(self, line_data: dict) -> ParsedResponse:
        custom_id = line_data.get("custom_id")
        result = line_data.get("result") or {}
        message_obj = result.get("message")

        if not isinstance(message_obj, dict):
            raise ValueError("Missing succeeded message payload in Anthropic batch line")

        parsed = self.parse_response(message_obj)
        parsed.custom_id = custom_id
        return parsed

    def _decode_anthropic_batch_error(self, line_data: dict) -> ParsedResponse:
        custom_id = line_data.get("custom_id")
        result = line_data.get("result") or {}
        result_type = result.get("type") or "errored"

        error_obj = result.get("error")
        if not isinstance(error_obj, dict):
            error_obj = {"type": result_type}

        error_message = error_obj.get("message") or error_obj.get("type") or result_type

        status_code = error_obj.get("status_code")
        if isinstance(status_code, int):
            error_code = status_code
        elif isinstance(status_code, str) and status_code.isdigit():
            error_code = int(status_code)
        else:
            error_code = 1

        return ParsedResponse(
            text=str(error_message),
            response_id=None,
            custom_id=custom_id,
            metadata=error_obj,
            error_code=error_code,
        )

    def decode_batch_content(self, content: str) -> List[BatchResult]:
        """Decode Anthropic JSONL batch output into BatchResult objects."""
        parsed_responses = []
        parsed_errors = []
        not_ok_i = []

        for line_i, line in enumerate(content.strip().split("\n")):
            if not line:
                continue
            try:
                line_data = json.loads(line)
                result_type = (line_data.get("result") or {}).get("type")

                if result_type == "succeeded":
                    parsed_responses.append(self._decode_anthropic_batch_success(line_data))
                else:
                    parsed_errors.append(self._decode_anthropic_batch_error(line_data))
                    not_ok_i.append(line_i)

            except json.JSONDecodeError as e:
                parsed_errors.append(
                    ParsedError(
                        text=f"JSON decode error: {str(e)}",
                        response_id=None,
                        custom_id="unknown",
                        metadata={},
                        error_code=1,
                    )
                )
                not_ok_i.append(line_i)

        return _split_batch_response(
            parsed_responses=parsed_responses,
            parsed_errors=parsed_errors,
            content=content,
            not_ok_i=not_ok_i,
        )

    def download_batch(
        self,
        batch_uuid: str,
        provider_type: str,
    ) -> List[BatchResult]:
        """Download Anthropic Message Batch results.

        :param batch_uuid: The UUID of the batch to download.
        :return: List of BatchResult objects containing the results and errors (if any).
            If nothing is ready yet, empty list is returned.
        """

        batch = self.client.messages.batches.retrieve(batch_uuid)
        if batch.processing_status != "ended":
            return []

        lines = []
        for item in self.client.messages.batches.results(batch_uuid):
            if isinstance(item, dict):
                lines.append(json.dumps(item))
            elif isinstance(item, str):
                lines.append(item)
            else:
                dump_json = getattr(item, "model_dump_json", None)
                if callable(dump_json):
                    lines.append(dump_json())
                else:
                    model_dump = getattr(item, "model_dump", None)
                    if callable(model_dump):
                        lines.append(json.dumps(model_dump(mode="json")))
                    else:
                        lines.append(json.dumps(item))

        content = "\n".join(lines)
        if not content:
            return []
        return self.decode_batch_content(content)
