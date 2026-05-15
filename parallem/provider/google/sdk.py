import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Union
from parallem.provider.base import (
    ConcurrentProvider,
    BaseProvider,
    BatchProvider,
    SyncProvider,
)
from parallem.provider.google.parser import GoogleParser
from parallem.types import (
    BatchResult,
    CommonQueryParameters,
    LLMIdentity,
    ParsedError,
    ParsedResponse,
)

if TYPE_CHECKING:
    from google import genai
    from pydantic import BaseModel

from parallem.utils._batch_helper import _split_batch_response


class GoogleProvider(BaseProvider):
    provider_type: str = "google"

    def __init__(self):
        self.parser = GoogleParser()

    def validate_request_compatibility(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> None:
        """Validate Google request compatibility.

        :param params: Common query parameters for the request.
        :return: None.
        """
        return None

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("gemini-2.5-flash", provider_type=self.provider_type)

    def parse_response(
        self, raw_response: Union["BaseModel", dict], provider_type: str = None
    ) -> ParsedResponse:
        """Parse Gemini API response into common format"""
        return self.parser.convert_response(raw_response, provider_type=provider_type)
        


class SyncGoogleProvider(SyncProvider, GoogleProvider):
    def __init__(self, client: "genai.Client"):
        super().__init__()
        self.client = client

    def prepare_sync_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a synchronous callable for Gemini API"""
        model_name, contents, config = self.parser.fix_config(params, **kwargs)
        return self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )


class ConcurrentGoogleProvider(ConcurrentProvider, GoogleProvider):
    def __init__(self, client: "genai.Client"):
        super().__init__()
        self.client = client

    def prepare_concurrent_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a concurrent coroutine for Gemini API"""
        model_name, contents, config = self.parser.fix_config(params, **kwargs)

        coro = self.client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        return coro



class BatchGoogleProvider(BatchProvider, GoogleProvider):
    def __init__(self, client: "genai.Client"):
        super().__init__()
        self.client = client

    def prepare_batch_call(
        self,
        params: CommonQueryParameters,
        custom_id: str,
        **kwargs,
    ) -> dict:
        """Convert CommonQueryParameters to Gemini batch request format"""
        body = self.parser.prepare_request(params, **kwargs)
        request = {
            "key": custom_id,
            "request": body,
        }

        return request

    def _decode_gemini_batch_result(
        self, result: dict, custom_id: str
    ) -> ParsedResponse:
        """Decode a single result from Gemini batch response"""

        # For successful responses, the result should be a GenerateContentResponse
        if "response" in result:
            response_data = result["response"]
            # Use existing parse_response method to handle the response
            parsed = self.parse_response(response_data)
            parsed.custom_id = custom_id
            return parsed
        else:
            # Fallback parsing
            raise ValueError("Unexpected gemini response format")

    def _decode_gemini_batch_error(
        self, result: dict, custom_id: str
    ) -> ParsedResponse:
        """Decode a single error from Gemini batch response"""

        error_info = result.get("error", {})
        error_message = error_info.get("message", "Unknown error")
        response_id = result.get("response_id", None)

        return ParsedResponse(
            text=error_message,
            response_id=response_id,
            custom_id=custom_id,
            metadata=error_info,
            error_code=1,  # TODO: figure out exact format of gemini
        )

    def get_batch_custom_ids(self, stuff, provider_type: str):
        custom_ids = []
        for item in stuff:
            if "key" not in item:
                raise ValueError("Each batch item must have a 'key' field.")
            custom_ids.append(item["key"])
        return custom_ids

    def submit_batch_to_provider(self, fpath: Path, llm: LLMIdentity) -> str:
        """
        Submit a batch of calls to the provider.

        This is called from the backend.
        """
        from google.genai import types

        num_lines = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for _ in f:
                num_lines += 1
        # Upload the file to Gemini File API
        uploaded_file = self.client.files.upload(
            file=fpath,
            config=types.UploadFileConfig(
                display_name=f"batch-requests-{num_lines}",
                mime_type="application/json",
            ),
        )

        # Create batch job
        batch_job = self.client.batches.create(
            model=llm.model_name,
            src=uploaded_file.name,
            config={
                "display_name": f"batch-job-{num_lines}-requests",
            },
        )
        return batch_job.name.removeprefix("batches/")

        # TODO: consider: a try/finally block option to clean up the temp files?

    def cancel_batch(self, batch_uuid: str, provider_type: str) -> None:
        """Cancel a Gemini batch job."""
        self.client.batches.cancel(name="batches/" + batch_uuid)

    def decode_batch_content(
        self,
        content: str,
    ) -> List[BatchResult]:
        """Decode content_str into dictionaries, with some error handling"""
        parsed_responses = []
        parsed_errors = []
        not_ok_i = []
        for line_i, line in enumerate(content.strip().split("\n")):
            if line:
                try:
                    line_data = json.loads(line)
                    custom_id = line_data.get("key", "unknown")

                    if "response" in line_data:
                        parsed_responses.append(
                            self._decode_gemini_batch_result(line_data, custom_id)
                        )
                    else:
                        parsed_errors.append(
                            self._decode_gemini_batch_error(line_data, custom_id)
                        )
                        not_ok_i.append(line_i)

                except json.JSONDecodeError as e:
                    # TODO: Handle malformed JSON lines
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
        """Download the results of a batch from the provider.

        :param batch_uuid: The UUID of the batch to download.
        :return: List of BatchResult objects containing the results and errors (if any).
            If nothing is ready yet, empty list is returned.
        """

        # Get batch job status
        batch_job = self.client.batches.get(name="batches/" + batch_uuid)

        # Check if job is completed
        completed_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }

        if batch_job.state.name not in completed_states:
            return []  # Still pending

        results = []

        if batch_job.state.name == "JOB_STATE_SUCCEEDED":
            if batch_job.dest and batch_job.dest.file_name:
                # Results are in a file
                try:
                    file_content = self.client.files.download(
                        file=batch_job.dest.file_name
                    )
                    content_str = file_content.decode("utf-8")

                    results.extend(self.decode_batch_content(content_str))
                except Exception as e:
                    results.append(
                        BatchResult(
                            status="error",
                            raw_output=str(e),
                            parsed_responses=None,
                        )
                    )
            else:
                raise ValueError("Batch job succeeded but no destination file found.")

        else:
            # Job failed, cancelled, or expired
            error_msg = getattr(
                batch_job, "error", f"Job state: {batch_job.state.name}"
            )
            results.append(
                BatchResult(
                    status="error",
                    raw_output=str(error_msg),
                    parsed_responses=None,
                )
            )

        return results
