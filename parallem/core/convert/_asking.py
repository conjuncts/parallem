from typing import TYPE_CHECKING, Callable, Optional, Union

from parallem.tools.auto_schema import to_tool_schema
from parallem.types import LLMIdentity, ServerTool

if TYPE_CHECKING:
    from parallem.provider.base import BaseProvider


def convert_tools(
    tools: Optional[list[Union[dict, ServerTool, Callable]]],
) -> Optional[list[Union[dict, ServerTool]]]:
    """
    Coerce tools into a consistent format: list of function call dicts or ServerTools
    Turns Callable into dict using pllm.to_tool_schema
    """
    # coerce callable tools to dict format with name and parameters
    if tools is not None:
        coerced_tools = []

        # if not sequence
        if not isinstance(tools, (list, tuple)):
            tools = [tools]
        for tool in tools:
            if isinstance(tool, (dict, ServerTool)):
                coerced_tools.append(tool)
            elif callable(tool):
                coerced_tools.extend(to_tool_schema(tool))
            else:
                raise ValueError(f"Tool {tool} is not a dict, ServerTool, or callable.")
        return coerced_tools
    return tools


def convert_options(
    llm,
    structured_output,
    kwargs,
    provider: "BaseProvider",
):
        """Helper method to ensure LLM options have the right type, coercing fields as needed."""
        # Handle legacy text_format alias
        legacy_text_format = kwargs.pop("text_format", None)
        if structured_output is not None and legacy_text_format is not None:
            raise ValueError(
                "Cannot specify both structured_output and text_format. "
                "text_format is a legacy alias for structured_output."
            )
        if structured_output is None:
            structured_output = legacy_text_format

        if llm is None:
            llm = provider.get_default_llm_identity()
        elif isinstance(llm, str):
            best_provider_type = provider.provider_type
            if best_provider_type in {"multi"}:
                best_provider_type = None
            llm = LLMIdentity(llm, provider_type=best_provider_type)

        provider_type = provider.provider_type
        if provider_type is None:
            provider_type = llm.provider_type

        return llm, provider_type, structured_output