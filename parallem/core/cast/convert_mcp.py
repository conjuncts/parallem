from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.types import Tool
    from mcp.types import ContentBlock
    from openai.types.responses.response_function_tool_call_output_item import ResponseFunctionToolCallOutputItem

def mcp_tool_to_openai_responses_tool(tool: "Tool") -> dict:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.inputSchema,
    }

def mcp_content_block_to_openai_fc_out(
    content_block: "ContentBlock"    
) -> "ResponseFunctionToolCallOutputItem":
    # Overall type:
    # see openai.types.responses.response_function_tool_call_output_item
    # ResponseFunctionToolCallOutputItem
    if content_block.type == "text":
        # "ResponseInputText"
        return {
            "type": "input_text",
            "text": content_block.text,
        }
    if content_block.type == "image":
        img_type = content_block.mimeType
        img_b64 = content_block.data
        # "ResponseInputImage"
        return {
            "type": "input_image",
            "image_url": f"data:{img_type};base64,{img_b64}"
            
        }
    if content_block.type == "resource":
        # "ResponseInputFile"
        # TODO untested
        resource = content_block.resource
        if getattr(resource, "blob"):
            content = resource.blob
        else:
            content = resource.text
        return {
            "type": "input_file",
            "file_data": content
        }
    return content_block.model_dump_json(exclude_none=True)