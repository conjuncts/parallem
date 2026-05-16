import os
from pathlib import Path
from typing import TYPE_CHECKING, List

from parallem.provider.base import BatchProvider
from parallem.provider.bedrock.sdk import BedrockProvider
from parallem.types import BatchResult, CommonQueryParameters, LLMIdentity

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
        request_params = self.parser.prepare_request(params, **kwargs)
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
        batch_client = boto3.client('bedrock', region_name=region) # not bedrock-runtime
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
        :param batch_uuid: The unique identifier for the batch to download
        """
        raise NotImplementedError

    def cancel_batch(self, batch_uuid: str, provider_type: str) -> None:
        """Cancel a batch on the provider.

        :param provider_type: Double check to make sure that batch_uuid is for the same provider.
        :param batch_uuid: The unique identifier for the batch to cancel.
        :return: None.
        """
        raise NotImplementedError