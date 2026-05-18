import json
from typing import TYPE_CHECKING, List, Union

from parallem.core.exception import ProviderCompatibilityError
from parallem.provider.base import BaseAdapter
from parallem.types import (
	CommonQueryParameters,
	FunctionCall,
	FunctionCallOutput,
	FunctionCallRequest,
	LLMDocument,
	MCPOutput,
	ParsedResponse,
	ServerTool,
)
from parallem.utils.image import get_type_and_b64, is_image

if TYPE_CHECKING:
	try:
		from mcp.types import ContentBlock
	except ImportError:
		pass


_IMAGE_MIME_TO_FORMAT = {
	"image/jpeg": "jpeg",
	"image/png": "png",
	"image/gif": "gif",
	"image/webp": "webp",
}


def _nova_image_block(image) -> dict:
	"""Convert a PIL Image into a Nova Invoke image block.

	:param image: PIL image instance.
	:return: Nova image content block.
	"""
	img_type, img_b64 = get_type_and_b64(
		image, allowed=["image/jpeg", "image/png", "image/gif", "image/webp"]
	)
	image_format = _IMAGE_MIME_TO_FORMAT.get(img_type, img_type.split("/", 1)[-1])
	return {
		"image": {
			"format": image_format,
			"source": {"bytes": img_b64},
		}
	}


def _fix_mcp_block(content_block: "ContentBlock") -> dict:
	"""Convert an MCP ContentBlock into a Nova tool result content block.

	:param content_block: MCP ContentBlock instance.
	:return: Nova ToolResultContentBlock dict.
	"""
	if content_block.type == "text":
		return {"text": content_block.text}
	if content_block.type == "image":
		image_format = _IMAGE_MIME_TO_FORMAT.get(
			content_block.mimeType, content_block.mimeType.split("/", 1)[-1]
		)
		return {
			"image": {
				"format": image_format,
				"source": {"bytes": content_block.data},
			}
		}
	if content_block.type == "resource":
		resource = content_block.resource
		try:
			blob = resource.blob
		except AttributeError:
			blob = None
		if blob is not None:
			return {"json": {"blob": blob}}
		try:
			text = resource.text
		except AttributeError:
			text = None
		if text is not None:
			return {"text": text}
	return {"text": content_block.model_dump_json(exclude_none=True)}


def _tool_result_content_from_output(content: object) -> list[dict]:
	"""Normalize tool outputs into Nova tool result content blocks.

	:param content: Tool output payload.
	:return: List of tool result content blocks.
	"""
	if content is None:
		return []
	if isinstance(content, list):
		if all(isinstance(item, dict) for item in content):
			return content
		return [{"text": str(item)} for item in content]
	if isinstance(content, dict):
		return [{"json": content}]
	if isinstance(content, bytes):
		return [{"text": content.decode("utf-8", errors="replace")}]
	return [{"text": str(content)}]


def _fix_docs_for_nova(documents: List[LLMDocument]) -> list[dict]:
	"""Convert LLM documents to Amazon Nova InvokeModel messages.

	:param documents: Input documents for the request.
	:return: A list of Nova message dicts.
	"""
	formatted_docs = []
	for doc in documents:
		if isinstance(doc, str):
			formatted_docs.append(
				{
					"role": "user",
					"content": [{"text": doc}],
				}
			)
			continue
		if isinstance(doc, FunctionCallRequest):
			content_blocks = []
			if doc.text_content:
				content_blocks.append({"text": doc.text_content})
			for call in doc.calls:
				content_blocks.append(
					{
						"toolUse": {
							"toolUseId": call.call_id,
							"name": call.name,
							"input": call.args,
						}
					}
				)
			formatted_docs.append(
				{
					"role": "assistant",
					"content": content_blocks,
				}
			)
			continue
		if isinstance(doc, (FunctionCallOutput, MCPOutput)):
			if isinstance(doc, MCPOutput):
				tool_content = [_fix_mcp_block(block) for block in doc.content]
			else:
				tool_content = _tool_result_content_from_output(doc.content)
			formatted_docs.append(
				{
					"role": "user",
					"content": [
						{
							"toolResult": {
								"toolUseId": doc.call_id,
								"content": tool_content,
								"status": "success",
							}
						}
					],
				}
			)
			continue
		if isinstance(doc, tuple) and len(doc) == 2:
			role, content = doc
			if role in {"system", "developer"}:
				role = "user"
			if role not in {"user", "assistant"}:
				raise ValueError(f"Unsupported role for Nova: {role}")
			formatted_docs.append(
				{
					"role": role,
					"content": [{"text": content}],
				}
			)
			continue
		if isinstance(doc, dict):
			if "role" in doc and "content" in doc:
				formatted_docs.append(doc)
				continue
		if is_image(doc):
			formatted_docs.append(
				{
					"role": "user",
					"content": [_nova_image_block(doc)],
				}
			)
			continue
		raise ValueError(f"Unsupported document type for Nova: {type(doc)}")
	return formatted_docs


def _prepare_tool_schema(
	func_schemas: List[Union[dict, ServerTool]],
) -> list[dict]:
	"""Convert tool definitions to Nova toolConfig schema.

	:param func_schemas: Tool schemas in OpenAI-style or Nova-style dicts.
	:return: List of Nova tool definitions.
	"""
	nova_tools: list[dict] = []
	for sch in func_schemas:
		if isinstance(sch, ServerTool):
			raise ProviderCompatibilityError(
				"Server tools are not supported for Amazon Nova InvokeModel."
			)

		if "toolSpec" in sch:
			nova_tools.append(sch)
			continue

		if "name" in sch and "inputSchema" in sch:
			nova_tools.append({"toolSpec": sch})
			continue

		if sch.get("type") == "function" and isinstance(sch.get("function"), dict):
			func = sch["function"]
		else:
			func = sch

		name = func.get("name")
		if not name:
			raise ValueError(f"Tool schema missing name: {sch}")
		description = func.get("description", "")
		parameters = func.get("parameters")
		if parameters is None and "input_schema" in func:
			parameters = func["input_schema"]

		if parameters is None:
			parameters = {"type": "object", "properties": {}, "required": []}

		nova_tools.append(
			{
				"toolSpec": {
					"name": name,
					"description": description,
					"inputSchema": {"json": parameters},
				}
			}
		)

	return nova_tools


def _collect_inference_config(model_kwargs: dict) -> dict:
	"""Collect Nova inferenceConfig from model kwargs.

	:param model_kwargs: Model kwargs to read and mutate.
	:return: Inference config dict.
	"""
	inference_config = model_kwargs.pop("inferenceConfig", None) or {}

	key_map = {
		"max_tokens": "maxTokens",
		"temperature": "temperature",
		"top_p": "topP",
		"top_k": "topK",
		"stop_sequences": "stopSequences",
		"reasoning_config": "reasoningConfig",
		"maxTokens": "maxTokens",
		"topP": "topP",
		"topK": "topK",
		"stopSequences": "stopSequences",
		"reasoningConfig": "reasoningConfig",
	}

	for raw_key, mapped_key in key_map.items():
		if raw_key in model_kwargs and mapped_key not in inference_config:
			inference_config[mapped_key] = model_kwargs.pop(raw_key)

	return inference_config


def _prepare_nova_body(
	params: CommonQueryParameters,
	model_kwargs: dict,
) -> tuple[str, Union[str, bytes, dict], dict]:
	"""Prepare the Amazon Nova InvokeModel request payload.

	:param params: Common query parameters for the request.
	:param model_kwargs: Model-specific keyword arguments.
	:return: A tuple of (model_name, body, invoke_options).
	"""
	nova_body = model_kwargs.pop("nova_body", None)
	invoke_options = model_kwargs.pop("nova_invoke_options", None) or model_kwargs.pop(
		"bedrock_invoke_options", None
	)
	if invoke_options is None:
		invoke_options = {}

	if nova_body is not None:
		if isinstance(nova_body, dict) and model_kwargs:
			return None, {**nova_body, **model_kwargs}, invoke_options
		if model_kwargs:
			raise ProviderCompatibilityError(
				"nova_body is not a dict; cannot merge model parameters."
			)
		return None, nova_body, invoke_options

	body: dict = {
		"messages": _fix_docs_for_nova(params["strict_documents"]),
	}

	instructions = params.get("instructions")
	if instructions:
		if "system" in model_kwargs:
			raise ProviderCompatibilityError(
				"Cannot supply both instructions and a system field in nova_body."
			)
		body["system"] = [{"text": instructions}]
	else:
		system_value = model_kwargs.pop("system", None)
		if system_value is not None:
			body["system"] = system_value

	tools = params.get("tools")
	tool_config = model_kwargs.pop("toolConfig", None)
	if tools:
		tool_config = tool_config or {}
		tool_config["tools"] = _prepare_tool_schema(tools)
		tool_choice = model_kwargs.pop("tool_choice", None)
		if tool_choice is None:
			tool_choice = model_kwargs.pop("toolChoice", None)
		if tool_choice is not None:
			tool_config["toolChoice"] = tool_choice
		body["toolConfig"] = tool_config
	elif tool_config is not None:
		body["toolConfig"] = tool_config

	inference_config = _collect_inference_config(model_kwargs)
	if inference_config:
		body["inferenceConfig"] = inference_config

	if model_kwargs:
		body.update(model_kwargs)

	return None, body, invoke_options


def _extract_text_from_nova_body(body: dict) -> tuple[str, list[FunctionCall]]:
	"""Extract response text and tool calls from a Nova response body.

	:param body: Parsed JSON response body.
	:return: A tuple of (text, function_calls).
	"""
	text = ""
	function_calls: list[FunctionCall] = []

	output = body.get("output")
	if isinstance(output, dict):
		message = output.get("message")
		if isinstance(message, dict):
			content = message.get("content")
			if isinstance(content, list):
				for block in content:
					if not isinstance(block, dict):
						continue
					if isinstance(block.get("text"), str):
						text += block["text"]
					tool_use = block.get("toolUse")
					if isinstance(tool_use, dict):
						name = tool_use.get("name")
						call_id = tool_use.get("toolUseId")
						arguments = tool_use.get("input")
						if name is not None and call_id is not None:
							function_calls.append(
								FunctionCall(
									name=name,
									arguments=arguments or {},
									call_id=call_id,
								)
							)

	return text, function_calls


def _convert_to_nova_response(raw_response: dict) -> ParsedResponse:
	if not isinstance(raw_response, dict):
		raise ValueError(f"Unsupported Nova response type: {type(raw_response)}")

	body_payload = raw_response.get("body", raw_response)
	try:
		raw_body = body_payload.read()
	except Exception:
		raw_body = body_payload

	if isinstance(raw_body, dict):
		body_obj = raw_body
	else:
		if isinstance(raw_body, bytes):
			raw_body_text = raw_body.decode("utf-8")
		else:
			raw_body_text = raw_body or ""

		try:
			body_obj = json.loads(raw_body_text) if raw_body_text else {}
		except json.JSONDecodeError:
			body_obj = {}

	text, function_calls = _extract_text_from_nova_body(body_obj)
	response_id = body_obj.get("id")
	if response_id is None:
		response_id = (raw_response.get("ResponseMetadata") or {}).get("RequestId")

	return ParsedResponse(
		text=text,
		response_id=response_id,
		metadata=body_obj,
		function_calls=function_calls or None,
	)


class NovaAdapter(BaseAdapter):
	def fix_docs(
		self,
		documents: List[LLMDocument],
	):
		return _fix_docs_for_nova(documents)

	def fix_tools(
		self,
		tools: List[Union[dict, ServerTool]],
	) -> list[dict]:
		return _prepare_tool_schema(tools)

	def fix_config(self, params, **kwargs):
		return _prepare_nova_body(params, kwargs)

	def convert_response(self, raw_response):
		return _convert_to_nova_response(raw_response)
