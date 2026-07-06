import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import httpx

from parallem.provider.base import (
    BaseProvider,
    BatchProvider,
    SyncProvider,
)
from parallem.provider.mistral.adapter import MistralAdapter
from parallem.provider.openai_chat.sdk import (
    AsyncOpenAIChatProvider,
    SyncOpenAIChatProvider,
)
from parallem.types import (
    BatchResult,
    CommonQueryParameters,
    LLMIdentity,
    ParsedError,
    ParsedResponse,
)
from parallem.utils._batch_helper import _split_batch_response

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI
    from pydantic import BaseModel


MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


class MistralProvider(BaseProvider):
    provider_type: str = "mistral"

    def __init__(self):
        super().__init__()
        self.adapter = MistralAdapter()

    def validate_request_compatibility(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> None:
        """Validate Mistral request compatibility.

        :param params: Common query parameters for the request.
        :return: None.
        """
        return None

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("mistral-large-latest", provider_type=self.provider_type)

    def parse_response(
        self, raw_response: Union["BaseModel", dict], llm: Optional[LLMIdentity] = None
    ) -> ParsedResponse:
        """Parse Mistral API response into common format."""
        return self.adapter.convert_response(raw_response)


class SyncMistralProvider(SyncProvider, MistralProvider):
    """Thin wrapper over OpenAI chat completions for Mistral.

    Mistral's API is OpenAI-compatible, so this delegates to
    :class:`SyncOpenAIChatProvider` for the actual API call logic.
    """

    def __init__(self, client: "OpenAI"):
        super().__init__()
        self.client = client
        self._inner = SyncOpenAIChatProvider(client)

    def prepare_sync_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a synchronous callable for Mistral API."""
        return self._inner.prepare_sync_call(params, **kwargs)


class AsyncMistralProvider(AsyncOpenAIChatProvider, MistralProvider):
    """Thin wrapper over OpenAI chat completions for Mistral.

    Mistral's API is OpenAI-compatible, so this delegates to
    :class:`AsyncOpenAIChatProvider` for the actual API call logic.
    """

    def __init__(self, client: "AsyncOpenAI"):
        super().__init__()
        self.client = client
        self._inner = AsyncOpenAIChatProvider(client)

    def prepare_async_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare an async coroutine for Mistral API."""
        return self._inner.prepare_async_call(params, **kwargs)


class BatchMistralProvider(BatchProvider, MistralProvider):
    """Mistral batch provider using the Mistral Files + Batch Jobs API.

    Uses httpx directly since Mistral's batch API differs from OpenAI's.
    Requires the ``MISTRAL_API_KEY`` environment variable.
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MISTRAL_API_KEY environment variable must be set "
                "to use the Mistral batch API."
            )
        self._http_client = httpx.Client(
            base_url=MISTRAL_BASE_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=300.0,
        )

    def prepare_batch_call(
        self,
        params: CommonQueryParameters,
        custom_id: str,
        **kwargs,
    ) -> dict:
        """Convert CommonQueryParameters to Mistral batch JSONL format.

        Mistral expects each JSONL line as::

            {"custom_id": "<id>", "body": {<request_body>}}
        """
        body = self.adapter.prepare_batch_request(params, **kwargs)
        return {
            "custom_id": custom_id,
            "body": body,
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
        """Submit a JSONL file as a Mistral batch job.

        1. Upload the file via ``POST /v1/files``.
        2. Create a batch job via ``POST /v1/batch/jobs``.

        Returns the batch job ID.
        """
        # 1. Upload the file
        with open(fpath, "rb") as f:
            files = {"file": (fpath.name, f, "application/jsonl")}
            data = {"purpose": "batch"}
            upload_resp = self._http_client.post("/files", data=data, files=files)
            upload_resp.raise_for_status()
            file_id = upload_resp.json()["id"]

        # 2. Create the batch job
        job_resp = self._http_client.post(
            "/batch/jobs",
            json={
                "input_files": [file_id],
                "model": llm.model_name,
                "endpoint": "/v1/chat/completions",
            },
        )
        job_resp.raise_for_status()
        return job_resp.json()["id"]

    def cancel_batch(self, batch_uuid: str, provider_type: str) -> None:
        """Cancel a Mistral batch job via ``POST /v1/batch/jobs/{id}/cancel``."""
        resp = self._http_client.post(f"/batch/jobs/{batch_uuid}/cancel")
        resp.raise_for_status()

    def download_batch(
        self,
        batch_uuid: str,
        provider_type: str,
    ) -> List[BatchResult]:
        """Download Mistral batch results.

        Returns an empty list if the batch is still pending.
        """
        # Get job status
        job_resp = self._http_client.get(f"/batch/jobs/{batch_uuid}")
        job_resp.raise_for_status()
        job = job_resp.json()

        status = job.get("status")

        # Terminal failure states
        if status in ("FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"):
            return [
                BatchResult(
                    status="error",
                    raw_output=f"Batch job ended with status: {status}",
                    parsed_responses=[
                        ParsedResponse(
                            text=status,
                            response_id=batch_uuid,
                            metadata=job,
                            error_code=1,
                        )
                    ],
                )
            ]

        if status != "SUCCESS":
            return []  # Still pending (QUEUED, RUNNING, CANCELLATION_REQUESTED)

        # Download output file
        output_file_id = job.get("output_file")
        if not output_file_id:
            return []

        content = self._download_file_content(output_file_id)
        if not content:
            return []

        return self.decode_batch_content(content)

    # -- internal helpers --------------------------------------------------

    def _download_file_content(self, file_id: str) -> str:
        """Download a file from the Mistral Files API."""
        resp = self._http_client.get(f"/files/{file_id}/content")
        resp.raise_for_status()
        return resp.text

    def decode_batch_content(self, content: str) -> List[BatchResult]:
        """Decode Mistral JSONL batch output into BatchResult objects."""
        parsed_responses: list[ParsedResponse] = []
        parsed_errors: list[ParsedResponse] = []
        not_ok_i: list[int] = []

        for line_i, line in enumerate(content.strip().split("\n")):
            if not line:
                continue
            try:
                line_data = json.loads(line)
                custom_id = line_data.get("custom_id")

                # Check for errors
                error = line_data.get("error")
                if error:
                    error_message = (
                        error.get("message")
                        or error.get("type")
                        or str(error)
                    )
                    parsed_errors.append(
                        ParsedResponse(
                            text=error_message,
                            response_id=None,
                            custom_id=custom_id,
                            metadata=error,
                            error_code=1,
                        )
                    )
                    not_ok_i.append(line_i)
                    continue

                # Success — locate the response body.
                # Mistral may return it as ``body``, ``output``, or the whole
                # line data may *be* the chat-completion response.
                body = (
                    line_data.get("body")
                    or line_data.get("output")
                    or line_data.get("outputs")
                    or line_data
                )
                if isinstance(body, list):
                    # ``outputs`` is sometimes a list with a single element
                    body = body[0] if body else {}

                parsed = self.parse_response(body)
                parsed.custom_id = custom_id
                parsed_responses.append(parsed)

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
