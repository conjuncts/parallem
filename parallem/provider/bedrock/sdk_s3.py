import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, List

from parallem.provider.base import BatchProvider
from parallem.provider.bedrock.sdk import BedrockProvider
from parallem.types import BatchResult, CommonQueryParameters, LLMIdentity, ParsedResponse
from parallem.utils._batch_helper import _split_batch_response

if TYPE_CHECKING:
    pass


class BatchBedrockProvider(BatchProvider, BedrockProvider):
    def prepare_batch_call(
        self,
        params: CommonQueryParameters,
        custom_id: str,
        **kwargs,
    ) -> dict:
        """
        Prepare batch call data for the backend to bookkeep.

        :param params: Common query parameters containing instructions, documents, llm, etc.
        :return: A dict/object representing the batch request format for this provider
        """
        request_params = self.adapter.prepare_batch_request(params, **kwargs)
        return {
            "recordId": custom_id,
            "modelInput": request_params,
        }

    def get_batch_custom_ids(self, stuff: list[dict], provider_type: str) -> list[str]:
        """Get batch IDs from dicts."""
        custom_ids = []
        for item in stuff:
            if "recordId" not in item:
                raise ValueError("Each batch item must have a 'recordId' field.")
            custom_ids.append(item["recordId"])
        return custom_ids

    def submit_batch_to_provider(self, fpath: Path, llm: LLMIdentity) -> str:
        """Submit a batch of calls to the provider."""
        # 1. Validate. Needs to have >= 100 lines
        if not fpath.exists():
            raise ValueError(f"File {fpath} does not exist.")
        if fpath.suffix != ".jsonl":
            raise ValueError(f"File {fpath} must be a .jsonl file.")
        line_ctr = 0
        with open(fpath, "r") as f:
            for line in f:
                if line.strip():
                    line_ctr += 1
        if line_ctr < 100:
            raise ValueError("Batch must contain at least 100 lines for Bedrock batch processing.")

        # 2. Upload .jsonl to S3
        import boto3

        bucket = os.environ.get("BEDROCK_S3_BUCKET")
        region = os.environ.get("AWS_REGION")
        s3 = boto3.client("s3", region_name=region)
        s3.upload_file(fpath, bucket, f".pllm/inputs/{fpath.name}")

        # 3. Submit batch job to Bedrock with S3 URI and return job ID
        batch_client = boto3.client("bedrock", region_name=region)  # not bedrock-runtime
        role_arn = os.environ.get("BEDROCK_ROLE_ARN")
        input_s3_uri = f"s3://{bucket}/.pllm/inputs/{fpath.name}"
        output_s3_uri = f"s3://{bucket}/.pllm/outputs/{fpath.stem}/"
        resp = batch_client.create_model_invocation_job(
            jobName="myJobName",
            modelId="google.gemma-3-4b-it",
            roleArn=role_arn,
            inputDataConfig={
                "s3InputDataConfig": {
                    "s3InputFormat": "JSONL",
                    "s3Uri": input_s3_uri,
                }
            },
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": output_s3_uri,
                }
            },
        )
        job_arn = resp["jobArn"]
        return job_arn

    def download_batch(self, batch_uuid: str, provider_type: str) -> List[BatchResult]:
        """Download the results of a batch from the provider.

        The list can contain both ready and error results.
        Empty list = still pending.
        - batch_status is one of "pending", "ready", or "error".

        :param provider_type: Double check to make sure that batch_uuid is for the same provider.
        :param batch_uuid: The unique identifier for the batch to download (job ARN for Bedrock).
        """
        import boto3

        region = os.environ.get("AWS_REGION")
        bucket = os.environ.get("BEDROCK_S3_BUCKET")

        # 1. Check job status
        batch_client = boto3.client("bedrock", region_name=region)
        job = batch_client.get_model_invocation_job(jobIdentifier=batch_uuid)
        status = job.get("status", "")

        # Bedrock statuses: Submitted, InProgress, Completed, Failed, Stopping, Stopped, Expired, PartiallyCompleted
        if status in ("Submitted", "InProgress", "Stopping"):
            return []  # Still pending

        if status in ("Failed", "Stopped", "Expired"):
            # The whole job failed before producing any output — surface one error BatchResult.
            return [
                BatchResult(
                    status="error",
                    raw_output=None,
                    parsed_responses=[
                        ParsedResponse(
                            text=f"Bedrock batch job ended with status '{status}'.",
                            response_id=None,
                            metadata={"job_status": status, "job_arn": batch_uuid},
                            error_code=1,
                        )
                    ],
                    location=None,
                )
            ]

        # status is Completed or PartiallyCompleted — download output JSONL from S3.
        # Derive the output prefix from the job's own outputDataConfig so we never
        # drift from what was actually submitted.
        output_config = job.get("outputDataConfig", {}).get("s3OutputDataConfig", {})
        output_s3_uri = output_config.get("s3Uri", "")

        # Parse bucket and prefix from the URI (s3://bucket/prefix/)
        if output_s3_uri.startswith("s3://"):
            parts = output_s3_uri[len("s3://") :].split("/", 1)
            output_bucket = parts[0]
            output_prefix = parts[1] if len(parts) > 1 else ""
        else:
            output_bucket = bucket
            output_prefix = ""

        s3 = boto3.client("s3", region_name=region)

        # List all objects under the output prefix to find result file(s).
        # Bedrock appends `.out` to the input filename, e.g. input.jsonl → input.jsonl.out
        paginator = s3.get_paginator("list_objects_v2")
        output_keys = []
        for page in paginator.paginate(Bucket=output_bucket, Prefix=output_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".jsonl.out") or key.endswith(".jsonl"):
                    output_keys.append(key)

        if not output_keys:
            # Job says it's done but S3 output isn't there yet — still treat as pending.
            return []

        # Collect all raw lines across every output file, then decode in one pass.
        all_lines: List[str] = []
        for key in output_keys:
            obj = s3.get_object(Bucket=output_bucket, Key=key)
            body = obj["Body"].read().decode("utf-8")
            all_lines.extend(body.splitlines())

        raw_output = "\n".join(all_lines)
        if not raw_output.strip():
            return []

        # 2. Decode the collected JSONL into BatchResult objects.
        return self.decode_batch_content(raw_output)

    def decode_batch_content(self, content: str) -> List[BatchResult]:
        """Decode Bedrock JSONL batch output into BatchResult objects.

        Each output line has the shape:
            {"recordId": "...", "modelOutput": {...}}   # success
            {"recordId": "...", "error": {...}}          # per-record error
        """
        parsed_responses: List[ParsedResponse] = []
        parsed_errors: List[ParsedResponse] = []
        not_ok_i: List[int] = []

        for line_i, line in enumerate(content.strip().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                parsed_errors.append(
                    ParsedResponse(
                        text=f"JSON decode error: {exc}",
                        response_id=None,
                        custom_id="unknown",
                        metadata={},
                        error_code=1,
                    )
                )
                not_ok_i.append(line_i)
                continue

            record_id = record.get("recordId", "")
            error_info = record.get("error")
            model_output = record.get("modelOutput")

            if error_info:
                error_message = (
                    error_info.get("message") or error_info.get("type") or str(error_info)
                )
                status_code = error_info.get("status_code") or error_info.get("statusCode")
                if isinstance(status_code, int):
                    error_code = status_code
                elif isinstance(status_code, str) and status_code.isdigit():
                    error_code = int(status_code)
                else:
                    error_code = 1

                parsed_errors.append(
                    ParsedResponse(
                        text=str(error_message),
                        response_id=None,
                        custom_id=record_id,
                        metadata=error_info
                        if isinstance(error_info, dict)
                        else {"raw": str(error_info)},
                        error_code=error_code,
                    )
                )
                not_ok_i.append(line_i)
            else:
                parsed = self.adapter.parse_response(model_output)
                parsed.custom_id = record_id
                parsed_responses.append(parsed)

        return _split_batch_response(
            parsed_responses=parsed_responses,
            parsed_errors=parsed_errors,
            content=content,
            not_ok_i=not_ok_i,
        )

    def cancel_batch(self, batch_uuid: str, provider_type: str) -> None:
        """Cancel a batch on the provider.

        :param provider_type: Double check to make sure that batch_uuid is for the same provider.
        :param batch_uuid: The unique identifier for the batch to cancel (job ARN for Bedrock).
        :return: None.
        """
        import boto3

        region = os.environ.get("AWS_REGION")
        batch_client = boto3.client("bedrock", region_name=region)

        # Bedrock only allows stopping jobs that are in a stoppable state.
        # Check first to avoid raising on already-terminal jobs.
        job = batch_client.get_model_invocation_job(jobIdentifier=batch_uuid)
        status = job.get("status", "")

        stoppable_statuses = {"Submitted", "InProgress"}
        if status not in stoppable_statuses:
            # Already in a terminal or stopping state — nothing to do
            return

        batch_client.stop_model_invocation_job(jobIdentifier=batch_uuid)
