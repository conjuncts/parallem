import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import httpx

from parallem.provider.base import (
    BaseAdapter,
    BaseProvider,
    BatchProvider,
    SyncProvider,
)
from parallem.provider.openai_chat.sdk import (
    AsyncOpenAIChatProvider,
    SyncOpenAIChatProvider,
    _fix_docs_for_openai_chat,
    _fix_tools_for_openai_chat,
    _prepare_response_format,
)
from parallem.types import (
    BatchResult,
    CommonQueryParameters,
    FunctionCall,
    LLMDocument,
    LLMIdentity,
    ParsedError,
    ParsedResponse,
    ServerTool,
)
from parallem.utils._batch_helper import _split_batch_response
from parallem.utils._quick_pydantic import is_pydantic_model

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )
    from pydantic import BaseModel


MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


class MistralAdapter(BaseAdapter):
    """Adapter for Mistral AI API (OpenAI-compatible chat completions).

    Mistral's API is compatible with OpenAI's chat completions format, so this
    adapter reuses the OpenAI chat completions document and tool formatting.
    """

    def prepare_sdk_request(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        raise NotImplementedError(
            "MistralAdapter does not implement prepare_sdk_request; "
            "use prepare_docs/prepare_tools directly instead."
        )

    def prepare_docs(
        self,
        documents: List[LLMDocument],
        instructions: Optional[str] = None,
    ) -> "List[ChatCompletionMessageParam]":
        return _fix_docs_for_openai_chat(documents, instructions)

    def prepare_tools(
        self,
        tools: List[Union[dict, ServerTool]],
    ):
        """Make tools ready for API calls."""
        return _fix_tools_for_openai_chat(tools)

    def fix_structured_output(
        self,
        structured_output: object,
    ) -> dict:
        return _prepare_response_format(structured_output)

    def prepare_batch_request(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> dict:
        """Prepare Mistral API request parameters from common query parameters."""
        instructions = params["instructions"]
        fixed_documents = self.prepare_docs(params["strict_documents"], instructions)
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self.prepare_tools(params.get("tools"))

        if structured_output is not None:
            if "response_format" in kwargs:
                raise AssertionError("Cannot supply both structured_output and response_format")
            kwargs["response_format"] = self.fix_structured_output(structured_output)

        body = {
            "model": llm.model_name,
            "messages": fixed_documents,
            "tools": tools,
            **kwargs,
        }
        return body

    def convert_response(
        self,
        raw_response: Union["BaseModel", dict],
    ) -> ParsedResponse:
        """Parse Mistral API response into common format."""

        def _parse_choice(choice: dict, texts: list[str], calls: list[FunctionCall]) -> None:
            message = choice.get("message") or {}
            content = message.get("content")
            if content:
                texts.append(content)

            tool_calls = message.get("tool_calls") or []
            for tool_call in tool_calls:
                if tool_call.get("type") == "function":
                    function_obj = tool_call.get("function") or {}
                    calls.append(
                        FunctionCall(
                            name=function_obj.get("name"),
                            arguments=function_obj.get("arguments"),
                            fcall_id=tool_call.get("id"),
                        )
                    )

        if isinstance(raw_response, dict):
            choices = raw_response.get("choices") or []
            texts: list[str] = []
            function_calls: list[FunctionCall] = []
            for choice in choices:
                _parse_choice(choice, texts, function_calls)
            text = "".join(texts)

            resp_id = raw_response.get("id")
            parsed_metadata = raw_response
        elif is_pydantic_model(raw_response):
            response: "ChatCompletion" = raw_response
            obj = response.model_dump(mode="json")
            resp_id = obj.get("id")

            choices = obj.get("choices") or []
            texts = []
            function_calls = []
            for choice in choices:
                _parse_choice(choice, texts, function_calls)
            text = "".join(texts)

            parsed_metadata = obj
        else:
            raise ValueError(f"Unsupported response type: {type(raw_response)}")

        return ParsedResponse(
            text=text,
            response_id=resp_id,
            metadata=parsed_metadata,
            function_calls=function_calls,
        )


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
            # Still pending (QUEUED, RUNNING, CANCELLATION_REQUESTED)
            return [
                BatchResult(
                    status="pending",
                    raw_output=None,
                    parsed_responses=None,
                    provider_status=status,
                    completed_count=job.get("completed_requests"),
                    total_count=job.get("total_requests"),
                )
            ]

        # Download output file
        output_file_id = job.get("output_file")
        if not output_file_id:
            return [
                BatchResult(
                    status="pending",
                    raw_output=None,
                    parsed_responses=None,
                    provider_status=status,
                    completed_count=job.get("completed_requests"),
                    total_count=job.get("total_requests"),
                )
            ]

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
