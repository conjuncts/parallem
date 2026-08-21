import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from parallem.core.exception import ProviderCompatibilityError
from parallem.types import BatchResult, LLMIdentity, ParsedError, ParsedResponse, ServerTool
from parallem.utils._batch_helper import _split_batch_response

if TYPE_CHECKING:
    from openai import OpenAI


def map_server_tools(
    tools: Optional[list[Union[dict, ServerTool]]],
    *,
    web_search_supported: bool = True,
) -> list[Union[dict, ServerTool]]:
    """Translate ServerTool into OpenAI tool dictionaries.

    :param tools: Tools to translate.
    :param web_search_supported: Web search is not supported in OpenAI ChatCompletions.
    :return: List of tool dictionaries.
    """
    if tools is None:
        return []

    openai_tools = []
    for tool in tools:
        if isinstance(tool, ServerTool):
            if tool.server_tool_type == "web_search":
                if not web_search_supported:
                    raise ProviderCompatibilityError(
                        "Web search tool is not supported with OpenAI ChatCompletions. Use Responses API instead."
                    )
                openai_tools.append({"type": "web_search", **tool.kwargs})
            elif tool.server_tool_type == "code_interpreter":
                openai_tools.append({"type": "code_interpreter", **tool.kwargs})
            elif tool.server_tool_type == "mcp":
                openai_tools.append(
                    {
                        "type": "mcp",
                        "server_label": tool.server_label,
                        "server_description": tool.server_description,
                        "server_url": tool.server_url,
                        "require_approval": tool.require_approval,
                    }
                )
            else:
                raise ValueError(f"Unsupported ServerTool type for OpenAI: {tool.server_tool_type}")
        else:
            openai_tools.append(tool)
    return openai_tools


class OpenAIBatchMixin:
    """Handles OpenAI Batch API for both Responses and ChatCompletions APIs."""

    batch_endpoint: str
    client: "OpenAI"

    def _decode_openai_batch_result(self, result: dict) -> ParsedResponse:
        """Decode a single result from OpenAI batch response.

        :param result: Batch result entry.
        :return: Parsed response.
        """
        custom_id = result["custom_id"]
        body = result["response"]["body"]

        parsed = self.parse_response(body)
        parsed.custom_id = custom_id
        return parsed

    def _decode_openai_batch_error(self, result: dict) -> ParsedResponse:
        """Decode a single error from OpenAI batch response.

        The text field contains the error code.

        :param result: Batch result entry.
        :return: Parsed response.
        """
        custom_id = result.pop("custom_id", None)

        err_obj = result.get("error", {})
        resp_obj = result.get("response", {})
        resp_id = (resp_obj.get("body") or {}).get("id", None)
        error_code = resp_obj.get("status_code")
        if error_code is not None:
            error_code = str(error_code)

        if err_obj is not None:
            return ParsedResponse(
                text=error_code or "",
                response_id=resp_id,
                custom_id=custom_id,
                metadata=err_obj,
            )

        body = resp_obj.get("body", {})
        body_error = body.get("error", {})
        return ParsedResponse(
            text=error_code or "",
            response_id=resp_id,
            custom_id=custom_id,
            metadata=body_error,
        )

    def get_batch_custom_ids(self, stuff: list[dict], provider_type: str) -> list[str]:
        """Extract custom ids from batch items.

        :param stuff: Batch items.
        :param provider_type: Provider type.
        :return: List of custom ids.
        """
        custom_ids = []
        for s in stuff:
            if not s.get("custom_id"):
                raise ValueError("Missing custom_id in batch item")
            custom_ids.append(s["custom_id"])
        return custom_ids

    def submit_batch_to_provider(self, fpath: Path, llm: LLMIdentity) -> str:
        """Submit a batch of calls to the provider.

        Returns the uuid.

        This is called from the backend.

        :param fpath: Path to batch input file.
        :param llm: LLM identity.
        :return: Batch UUID.
        """
        with open(fpath, "rb") as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")
            batch_input_file_id = batch_input_file.id

        batch_obj = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint=self.batch_endpoint,
            completion_window="24h",
        )

        return batch_obj.id

    def cancel_batch(self, batch_uuid: str, provider_type: str) -> None:
        """Cancel a batch on OpenAI.

        :param batch_uuid: Batch UUID.
        :param provider_type: Provider type.
        :return: None.
        """
        self.client.batches.cancel(batch_uuid)

    def decode_batch_content(self, content: str) -> List[BatchResult]:
        """Decode content_str into dictionaries, with some error handling.

        :param content: Batch output content.
        :return: List of batch results.
        """
        parsed_responses = []
        parsed_errors = []
        not_ok_i = []
        for line_i, line in enumerate(content.strip().split("\n")):
            if not line:
                continue
            try:
                line_data = json.loads(line)

                response = line_data.get("response", {})
                status_code = response.get("status_code")
                has_error = line_data.get("error") or (status_code and status_code != 200)

                if not has_error and status_code == 200:
                    parsed_responses.append(self._decode_openai_batch_result(line_data))
                else:
                    decoded_err = self._decode_openai_batch_error(line_data)
                    decoded_err.error_code = status_code
                    parsed_errors.append(decoded_err)
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
        """Download the results of a batch from the provider.

        :param batch_uuid: The UUID of the batch to download.
        :param provider_type: Provider type.
        :return: List of BatchResult objects containing the results and errors (if any).
            If nothing is ready yet, empty list is returned.
        """
        batch = self.client.batches.retrieve(batch_uuid)
        err_file_id = batch.error_file_id
        out_file_id = batch.output_file_id

        if batch.errors:
            print(f"Batch {batch_uuid} failed with errors: {batch.errors}")
            return [
                BatchResult(
                    status="error",
                    raw_output=str(batch.errors),
                    parsed_responses=None,
                )
            ]

        _created_at = batch.created_at
        _completed_at = batch.completed_at

        if out_file_id is None and err_file_id is None:
            counts = batch.request_counts
            return [
                BatchResult(
                    status="pending",
                    raw_output=None,
                    parsed_responses=None,
                    provider_status=batch.status,
                    completed_count=counts.completed if counts is not None else None,
                    total_count=counts.total if counts is not None else None,
                )
            ]

        results = []

        if out_file_id is not None:
            out_content = self.client.files.content(out_file_id).text
            results.extend(self.decode_batch_content(out_content))

        if err_file_id is not None:
            err_content = self.client.files.content(err_file_id).text

            try:
                parsed_errors = [
                    self._decode_openai_batch_error(json.loads(line))
                    for line in err_content.strip().split("\n")
                    if line
                ]
                err_res = BatchResult(
                    status="error",
                    raw_output=err_content,
                    parsed_responses=parsed_errors,
                )
            except json.JSONDecodeError:
                err_res = BatchResult(
                    status="error",
                    raw_output=err_content,
                    parsed_responses=None,
                )
            results.append(err_res)
        return results
