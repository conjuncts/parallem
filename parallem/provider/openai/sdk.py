from typing import TYPE_CHECKING, List, Union

from parallem.provider.openai.common import OpenAIBatchMixin, map_server_tools
from parallem.provider.base import (
    ConcurrentProvider,
    BaseProvider,
    BatchProvider,
    SyncProvider,
)
from parallem.provider.openai.openai_tools import to_strict_json_schema
from parallem.types import (
    CommonQueryParameters,
    FunctionCallRequest,
    FunctionCallOutput,
    LLMDocument,
    LLMIdentity,
    ParsedResponse,
    ServerTool,
    FunctionCall,
)
from parallem.utils._quick_pydantic import is_pydantic_model
from parallem.utils.image import get_type_and_b64, is_image

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI
    from openai.types.responses.response_input_param import Message
    from openai.types.responses.response import Response
    from pydantic import BaseModel


class OpenAIProvider(BaseProvider):
    provider_type: str = "openai"

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

    def _fix_docs_for_openai(
        self,
        documents: List[LLMDocument],
    ) -> "List[Message]":
        """Ensure documents are in the correct format for OpenAI API"""

        formatted_docs = []
        for doc in documents:
            if isinstance(doc, str):
                msg: "Message" = {
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
                            "call_id": call.call_id,
                            "type": "function_call",
                        }
                    )
            elif isinstance(doc, FunctionCallOutput):
                msg = {
                    "type": "function_call_output",
                    "call_id": doc.call_id,
                    "output": str(doc.content),
                }
                formatted_docs.append(msg)
            elif isinstance(doc, tuple) and len(doc) == 2:
                # Handle Tuple[Literal["user", "assistant", "system", "developer"], str]
                # Valid roles for OpenAI are:
                # from openai.types.responses.response_input_param import Message
                # from openai.types.responses.response_output_message_param import ResponseOutputMessageParam
                role, content = doc

                msg: "Message" = {
                    "role": role,
                    "content": content,
                }
                formatted_docs.append(msg)
            elif is_image(doc):
                img_type, img_b64 = get_type_and_b64(
                    doc, allowed=["image/jpeg", "image/png", "image/gif", "image/webp"]
                )
                formatted_docs.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:{img_type};base64,{img_b64}",
                            },
                        ],
                    }
                )
            else:
                raise ValueError(f"Unsupported document type: {type(doc)}")
        return formatted_docs

    def _fix_server_tools_for_openai(
        self,
        tools: list[Union[dict, ServerTool]],
    ):
        """Translate ServerTool into OpenAI API format"""
        return map_server_tools(tools)

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("gpt-5-nano", provider_type=self.provider_type)

    def parse_response(
        self, raw_response: Union["BaseModel", dict], provider_type: str = None
    ) -> ParsedResponse:
        """Parse OpenAI API response into common format"""
        if isinstance(raw_response, dict):
            # Dict response (e.g., from batch API)

            if "output_text" in raw_response:
                # Used in testing
                text = raw_response["output_text"]
                function_calls = []
            else:
                # Extract text from OpenAI responses API format
                function_calls = []
                texts: List[str] = []
                for output in raw_response.get("output", []):
                    if output["type"] == "function_call":
                        function_calls.append(
                            FunctionCall(
                                name=output["name"],
                                arguments=output["arguments"],
                                call_id=output.get("call_id"),
                            )
                        )
                    elif output["type"] == "custom_tool_call":
                        function_calls.append(
                            FunctionCall(
                                name=output["name"],
                                arguments=output["input"],
                                call_id=output.get("call_id"),
                            )
                        )
                    elif output["type"] == "message":
                        for content in output["content"]:
                            if content["type"] == "output_text":
                                texts.append(content["text"])
                text = "".join(texts)

            resp_id = raw_response.pop("id", None)
            parsed_metadata = raw_response
        elif is_pydantic_model(raw_response):
            # Pydantic model (e.g., from openai.types.responses.response.Response)
            response: Response = raw_response
            text = response.output_text
            obj = response.model_dump(mode="json")
            resp_id = response.id
            obj.pop("id", None)

            function_calls = []
            for item in response.output:
                if item.type == "function_call":
                    function_calls.append(
                        FunctionCall(
                            name=item.name,
                            arguments=item.arguments,
                            call_id=item.call_id,
                        )
                    )
                elif item.type == "custom_tool_call":
                    function_calls.append(
                        FunctionCall(
                            name=item.name, arguments=item.input, call_id=item.call_id
                        )
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


class SyncOpenAIProvider(SyncProvider, OpenAIProvider):
    def __init__(self, client: "OpenAI"):
        self.client = client

    def prepare_sync_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a synchronous callable for OpenAI API"""
        instructions = params["instructions"]
        fixed_documents = self._fix_docs_for_openai(params["strict_documents"])
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self._fix_server_tools_for_openai(params.get("tools"))

        if structured_output is not None:
            return self.client.responses.parse(
                model=llm.model_name,
                instructions=instructions,
                input=fixed_documents,
                text_format=structured_output,
                tools=tools,
                **kwargs,
            )

            # if "text" not in kwargs:
            #     kwargs["text"] = {}

            # schema = to_strict_json_schema(structured_output)
            # kwargs["text"]["format"] = {
            #     "type": "json_schema",
            #     "strict": True,
            #     "name": schema.get("title", "UnknownSchema"),
            #     "schema": schema,
            # }
        return self.client.responses.create(
            model=llm.model_name,
            instructions=instructions,
            input=fixed_documents,
            tools=tools,
            **kwargs,
        )


class ConcurrentOpenAIProvider(ConcurrentProvider, OpenAIProvider):
    def __init__(self, client: "AsyncOpenAI"):
        self.client = client

    def prepare_concurrent_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a concurrent coroutine for OpenAI API"""
        instructions = params["instructions"]
        fixed_documents = self._fix_docs_for_openai(params["strict_documents"])
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self._fix_server_tools_for_openai(params.get("tools"))

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
        self.client = client

    def prepare_batch_call(
        self,
        params: CommonQueryParameters,
        custom_id: str,
        **kwargs,
    ):
        """Prepare batch call data for OpenAI"""
        instructions = params["instructions"]
        fixed_documents = self._fix_docs_for_openai(params["strict_documents"])
        llm = params["llm"]
        structured_output = params.get("structured_output")
        tools = self._fix_server_tools_for_openai(params.get("tools"))

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

        body = {
            "model": llm.model_name,
            "instructions": instructions,
            "input": fixed_documents,
            "tools": tools,
            **kwargs,
        }
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
