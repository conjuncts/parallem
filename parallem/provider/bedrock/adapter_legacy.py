from typing import TYPE_CHECKING, List, Union

from parallem.core.exception import ProviderCompatibilityError
from parallem.provider.base import (
    BaseAdapter,
)
from parallem.types import (
    CommonQueryParameters,
    FunctionCall,
    FunctionCallOutput,
    FunctionCallRequest,
    LLMDocument,
    ParsedResponse,
)

if TYPE_CHECKING:
    pass


def _fix_docs_for_bedrock(documents: List[LLMDocument]) -> list[dict]:
    """Convert LLM documents to Bedrock InvokeModel messages.

    :param documents: The input documents for the request.
    :return: A list of Bedrock message dicts.
    """
    formatted_docs = []
    for doc in documents:
        if isinstance(doc, str):
            formatted_docs.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": doc}],
                }
            )
            continue
        if isinstance(doc, FunctionCallRequest):
            raise ProviderCompatibilityError(
                "Bedrock InvokeModel does not support tool call requests."
            )
        if isinstance(doc, FunctionCallOutput):
            raise ProviderCompatibilityError(
                "Bedrock InvokeModel does not support tool call outputs."
            )
        if isinstance(doc, tuple) and len(doc) == 2:
            role, content = doc
            if role in {"system", "developer"}:
                role = "user"
            if role not in {"user", "assistant"}:
                raise ValueError(f"Unsupported role for Bedrock: {role}")
            formatted_docs.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                }
            )
            continue
        if isinstance(doc, dict):
            if "role" in doc and "content" in doc:
                formatted_docs.append(doc)
                continue
        raise ValueError(f"Unsupported document type for Bedrock: {type(doc)}")
    return formatted_docs


def _prepare_bedrock_body(
    params: CommonQueryParameters,
    model_kwargs: dict,
) -> tuple[str, Union[str, bytes, dict], dict]:
    """Prepare the Bedrock InvokeModel request payload.

    :param params: Common query parameters for the request.
    :param model_kwargs: Model-specific keyword arguments.
    :return: A tuple of (model_name, body, invoke_options).
    """
    bedrock_body = model_kwargs.pop("bedrock_body", None)
    model_family = model_kwargs.pop("bedrock_model_family", None)

    if bedrock_body is not None:
        if isinstance(bedrock_body, dict) and model_kwargs:
            return {**bedrock_body, **model_kwargs}
        if model_kwargs:
            raise ProviderCompatibilityError(
                "bedrock_body is not a dict; cannot merge model parameters."
            )
        return bedrock_body

    if model_family is None:
        model_family = "anthropic"

    if model_family != "anthropic":
        raise ProviderCompatibilityError(
            "Bedrock InvokeModel requires bedrock_body or bedrock_model_family='anthropic'."
        )

    messages = _fix_docs_for_bedrock(params["strict_documents"])
    body: dict = {
        "anthropic_version": model_kwargs.pop(
            "anthropic_version", "bedrock-2023-05-31"
        ),
        "messages": messages,
        **model_kwargs,
    }

    instructions = params.get("instructions")
    if instructions:
        if "system" in body:
            raise ProviderCompatibilityError(
                "Cannot supply both instructions and a system field in bedrock_body."
            )
        body["system"] = instructions

    return body


def _extract_text_from_bedrock_body(body: dict) -> tuple[str, list[FunctionCall]]:
    """Extract response text and tool calls from a Bedrock response body.

    :param body: Parsed JSON response body.
    :return: A tuple of (text, function_calls).
    """
    text = ""
    function_calls: list[FunctionCall] = []

    content = body.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                text += item["text"]
            if item.get("type") == "tool_use":
                name = item.get("name")
                call_id = item.get("id")
                arguments = item.get("input")
                if name is not None and call_id is not None:
                    function_calls.append(
                        FunctionCall(
                            name=name,
                            arguments=arguments or {},
                            call_id=call_id,
                        )
                    )
        if text or function_calls:
            return text, function_calls

    if isinstance(body.get("outputText"), str):
        return body["outputText"], function_calls

    results = body.get("results")
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict) and isinstance(first.get("outputText"), str):
            return first["outputText"], function_calls
        if isinstance(first, dict) and isinstance(first.get("text"), str):
            return first["text"], function_calls

    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"], function_calls
            if isinstance(first.get("text"), str):
                return first["text"], function_calls

    for key in ("completion", "generation", "generated_text", "output"):
        if isinstance(body.get(key), str):
            return body[key], function_calls

    return text, function_calls

def _convert_to_bedrock_response(raw_response: dict) -> ParsedResponse:
    if not isinstance(raw_response, dict):
        raise ValueError(f"Unsupported Bedrock response type: {type(raw_response)}")

    body_obj = raw_response

    text, function_calls = _extract_text_from_bedrock_body(body_obj)
    response_id = body_obj.get("id")
    if response_id is None:
        response_id = (raw_response.get("ResponseMetadata") or {}).get("RequestId")

    return ParsedResponse(
        text=text,
        response_id=response_id,
        metadata=body_obj,
        function_calls=function_calls or None,
    )

class BedrockLegacyAdapter(BaseAdapter):
    def fix_docs(
        self,
        documents: List[LLMDocument],
    ):
        return _fix_docs_for_bedrock(documents)

    def prepare_request(self, params, **kwargs):
        return _prepare_bedrock_body(params, kwargs)

    def convert_response(self, raw_response):
        return _convert_to_bedrock_response(raw_response)