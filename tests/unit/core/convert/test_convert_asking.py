
import pytest

from parallem.core.convert._asking import convert_tools


class TestAskLLMConversion:
    """Test coercion methods in AgentContext"""

    def test_coerce_tools_single_dict(self):
        tool = {"name": "test_tool", "parameters": {}}
        coerced = convert_tools(tool)
        assert coerced == [tool]

    def test_coerce_tools_list_mixed(self):

        def my_tool(x: int):
            """My tool description"""
            return x

        dict_tool = {"name": "dict_tool", "parameters": {}}
        tools = [dict_tool, my_tool]

        coerced = convert_tools(tools)

        assert len(coerced) == 2
        assert coerced[0] == dict_tool
        # to_tool_schema returns a LIST of schemas
        tool_schema = coerced[1]
        assert tool_schema["type"] == "function"
        assert tool_schema["name"] == "my_tool"
        assert "parameters" in tool_schema

    def test_coerce_tools_invalid_type(self):
        with pytest.raises(ValueError, match="is not a dict, ServerTool, or callable"):
            convert_tools(123)

