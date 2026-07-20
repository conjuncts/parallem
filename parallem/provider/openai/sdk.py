import base64
from typing import TYPE_CHECKING, List, Optional, Union

from parallem.provider.openai.common import OpenAIBatchMixin, map_server_tools
from parallem.provider.openai.openai_tools import to_strict_json_schema
from parallem.provider.base import (
    AsyncProvider,
    BaseAdapter,
    BaseProvider,
    BatchProvider,
    SyncProvider,
)
from parallem.types import (
    CommonQueryParameters,
    FileInput,
    FunctionCall,
    FunctionCallOutput,
    FunctionCallRequest,
    LLMDocument,
    LLMIdentity,
    MCPOutput,
    MultipartDocument,
    ParsedResponse,
    ServerTool,
)
from parallem.utils._quick_pydantic import is_pydantic_model
from parallem.utils.image import is_image, is_image_url, to_image_url_str

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI
    from openai.types.responses.response_input_item_param import (
        ResponseInputItemParam,
        FunctionCallOutput as NativeFunctionCallOutput,
    )
    from openai.types.responses.easy_input_message_param import EasyInputMessageParam
    from openai.types.responses.response_input_param import Message
    from openai.types.responses.response import Response
    from openai.types.responses.response_function_tool_call_output_item import (
        ResponseFunctionToolCallOutputItem,
    )
    from openai.types.responses.response_input_file import ResponseInputFile
    from openai.types.responses.response_input_content_param import ResponseInputContentParam
    from pydantic import BaseModel

    from mcp.types import ContentBlock


def _fix_part_for_openai(
    doc: LLMDocument,
) -> "ResponseInputContentParam":
    if isinstance(doc, str):
        return {
            "type": "input_text",
            "text": doc,
        }
    if isinstance(doc, FileInput):
        if doc.file_url:
            input_file: "ResponseInputFile" = {
                "type": "input_file",
                "file_url": doc.file_url,
                **doc.kwargs
            }
        elif doc.file_content:
            b64 = base64.b64encode(doc.file_content).decode("utf-8")
            input_file: "ResponseInputFile" = {
                "type": "input_file",
                "filename": doc.filename,
                "file_data": f"data:{doc.mime_type};base64,{b64}",
                **doc.kwargs
            }
        else:
            raise ValueError("FileInput must have either file_url or file_content.")
        return input_file
    elif is_image(doc) or is_image_url(doc):

        as_image_url = to_image_url_str(
            doc, allowed_mimetypes=["image/jpeg", "image/png", "image/gif", "image/webp"]
        )
        return {
            "type": "input_image",
            "image_url": as_image_url
        }
    print("Warning: received unsupported document type for OpenAI API. Returning None.")
    return None

def _fix_docs_for_openai(
    documents: List[LLMDocument],
) -> "List[ResponseInputItemParam]":
    """Ensure documents are in the correct format for OpenAI API"""

    formatted_docs: List["ResponseInputItemParam"] = []
    for doc in documents:
        if isinstance(doc, str):
            msg: "EasyInputMessageParam" = {
                "role": "user",
                "content": doc,
            }
            formatted_docs.append(msg)
        elif isinstance(doc, FunctionCallRequest):
            if doc.text_content:
                formatted_docs.append(
                    {
                        "role": "assistant",
                        "content": doc.text_content,
                    }
                )
            for call in doc.calls:
                formatted_docs.append(
                    {
                        "name": call.name,
                        "arguments": call.arg_str,
                        "call_id": call.fcall_id,
                        "type": "function_call",
                    }
                )
        elif isinstance(doc, (FunctionCallOutput, MCPOutput)):
            fc_content = doc.content
            if isinstance(doc, MCPOutput):
                fc_content = [_fix_mcp_block(x) for x in fc_content]

            msg: "NativeFunctionCallOutput" = {
                "type": "function_call_output",
                "call_id": doc.fcall_id,
                "output": fc_content,
            }
            formatted_docs.append(msg)
        elif isinstance(doc, tuple) and len(doc) == 2:
            role, content = doc

            msg: "Message" = {
                "role": role,
                "content": content,
            }
            formatted_docs.append(msg)
        elif (
            isinstance(doc, FileInput)
            or is_image(doc)
            or is_image_url(doc)
        ):
            part = _fix_part_for_openai(doc)
            msg: "EasyInputMessageParam" = {
                "role": "user",
                "content": [part],
            }
            formatted_docs.append(msg)
        elif isinstance(doc, MultipartDocument):
            
            formatted_docs.append({
                "role": doc.role,
                "content": [
                    _fix_part_for_openai(part) for part in doc.parts
                ]
            })
        elif isinstance(doc, dict):
            # Assume it's already in the correct format for OpenAI API
            formatted_docs.append(doc)
        else:
            raise ValueError(f"Unsupported document type: {type(doc)}")
    return formatted_docs


def _fix_mcp_block(content_block: "ContentBlock") -> "ResponseFunctionToolCallOutputItem":
    if content_block.type == "text":
        return {
            "type": "input_text",
            "text": content_block.text,
        }
    if content_block.type == "image":
        return {
            "type": "input_image",
            "image_url": f"data:{content_block.mimeType};base64,{content_block.data}",
        }
    if content_block.type == "resource":
        resource = content_block.resource
        if getattr(resource, "blob", None):
            return {
                "type": "input_file",
                "file_data": resource.blob,
            }
        return {
            "type": "input_text",
            "text": resource.text,
        }
    return content_block.model_dump_json(exclude_none=True)


class OpenAIAdapter(BaseAdapter):
    """Parses OpenAI API responses into a common format for downstream processing."""

    def prepare_sdk_request(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        raise NotImplementedError(
            "OpenAIAdapter does not implement fix_config; use prepare_request instead."
        )

    def prepare_docs(
        self,
        documents: List[LLMDocument],
    ) -> "List[ResponseInputItemParam]":
        return _fix_docs_for_openai(documents)

    def prepare_tools(
        self,
        tools: List[Union[dict, ServerTool]],
    ):
        """Make tools ready for API calls."""
        return map_server_tools(tools)

    def prepare_batch_request(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> tuple:
        """Prepare OpenAI API request parameters from common query parameters."""
        instructions = params["instructions"]
        fixed_documents = self.prepare_docs(params["strict_documents"])
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self.prepare_tools(params.get("tools"))

        if structured_output is not None:
            if "text" not in kwargs:
                kwargs["text"] = {}

            assert not kwargs["text"].get("format"), (
                "Cannot supply both structured_output and text.format"
            )
            schema = to_strict_json_schema(structured_output)
            kwargs["text"]["format"] = {
                "type": "json_schema",
                "strict": True,
                "name": schema.get("title", "UnknownSchema"),
                "schema": schema,
            }

        return {
            "model": llm.model_name,
            "instructions": instructions,
            "input": fixed_documents,
            "tools": tools,
            **kwargs,
        }

    def convert_response(self, raw_response: Union["BaseModel", dict]) -> ParsedResponse:
        """Parse OpenAI API response into common format"""
        if isinstance(raw_response, dict):
            if "output_text" in raw_response:
                text = raw_response["output_text"]
                function_calls = []
            else:
                function_calls = []
                texts: List[str] = []
                for output in raw_response.get("output", []):
                    if output["type"] == "function_call":
                        function_calls.append(
                            FunctionCall(
                                name=output["name"],
                                arguments=output["arguments"],
                                fcall_id=output.get("call_id"),
                            )
                        )
                    elif output["type"] == "custom_tool_call":
                        function_calls.append(
                            FunctionCall(
                                name=output["name"],
                                arguments=output["input"],
                                fcall_id=output.get("call_id"),
                            )
                        )
                    elif output["type"] == "message":
                        for content in output["content"]:
                            if content["type"] == "output_text":
                                texts.append(content["text"])
                text = "".join(texts)

            resp_id = raw_response.get("id")
            parsed_metadata = raw_response
        elif is_pydantic_model(raw_response):
            response: Response = raw_response
            text = response.output_text
            obj = response.model_dump(mode="json")
            resp_id = response.id

            function_calls = []
            for item in response.output:
                if item.type == "function_call":
                    function_calls.append(
                        FunctionCall(
                            name=item.name,
                            arguments=item.arguments,
                            fcall_id=item.call_id,
                        )
                    )
                elif item.type == "custom_tool_call":
                    function_calls.append(
                        FunctionCall(name=item.name, arguments=item.input, fcall_id=item.call_id)
                    )

            parsed_metadata = obj
        else:
            raise ValueError(f"Unsupported response type: {type(raw_response)}")
        return ParsedResponse(
            text=text,
            response_id=resp_id,
            metadata=parsed_metadata,
            function_calls=function_calls,
        )


class OpenAIProvider(BaseProvider):
    provider_type: str = "openai"

    def __init__(self):
        self.adapter = OpenAIAdapter()

    def validate_request_compatibility(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ) -> None:
        """Validate OpenAI request compatibility.

        :param params: Common query parameters for the request.
        :return: None.
        """
        return None

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("gpt-5-nano", provider_type=self.provider_type)

    def parse_response(
        self, raw_response: Union["BaseModel", dict], llm: Optional[LLMIdentity] = None
    ) -> ParsedResponse:
        return self.adapter.convert_response(raw_response)


class SyncOpenAIProvider(SyncProvider, OpenAIProvider):
    def __init__(self, client: "OpenAI"):
        super().__init__()
        self.client = client

    def prepare_sync_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a synchronous callable for OpenAI API"""
        instructions = params["instructions"]
        fixed_documents = self.adapter.prepare_docs(params["strict_documents"])
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self.adapter.prepare_tools(params.get("tools"))

        if structured_output is not None:
            return self.client.responses.parse(
                model=llm.model_name,
                instructions=instructions,
                input=fixed_documents,
                text_format=structured_output,
                tools=tools,
                **kwargs,
            )

        return self.client.responses.create(
            model=llm.model_name,
            instructions=instructions,
            input=fixed_documents,
            tools=tools,
            **kwargs,
        )


class AsyncOpenAIProvider(AsyncProvider, OpenAIProvider):
    def __init__(self, client: "AsyncOpenAI"):
        super().__init__()
        self.client = client

    def prepare_async_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare an async coroutine for OpenAI API"""
        instructions = params["instructions"]
        fixed_documents = self.adapter.prepare_docs(params["strict_documents"])
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self.adapter.prepare_tools(params.get("tools"))

        if structured_output is not None:
            coro = self.client.responses.parse(
                model=llm.model_name,
                instructions=instructions,
                input=fixed_documents,
                text_format=structured_output,
                tools=tools,
                **kwargs,
            )
        else:
            coro = self.client.responses.create(
                model=llm.model_name,
                instructions=instructions,
                input=fixed_documents,
                tools=tools,
                **kwargs,
            )

        return coro


class BatchOpenAIProvider(OpenAIBatchMixin, BatchProvider, OpenAIProvider):
    batch_endpoint = "/v1/responses"

    def __init__(self, client: "OpenAI"):
        super().__init__()
        self.client = client

    def prepare_batch_call(
        self,
        params: CommonQueryParameters,
        custom_id: str,
        **kwargs,
    ) -> dict:
        """Prepare batch call data for OpenAI"""
        body = self.adapter.prepare_batch_request(params, **kwargs)
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
