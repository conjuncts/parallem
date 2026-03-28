# Tools

To see the diverse formats that need support:


```python
# openai
# https://platform.openai.com/docs/guides/function-calling#custom-tools
computed_tool_output = {
    "type": "function_call_output",
    "call_id": function_calls[0][2],
    "output": ls_tool(function_calls[0][1]),
}

# anthropic
# https://docs.claude.com/en/docs/agents-and-tools/tool-use/implement-tool-use
computed_tool_output = {
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": function_calls[0][2],
            "output": ls_tool(function_calls[0][1]),
        },
        {
            "type": "text",
            "text": "What should I do next?",
        },  # ✅ Text after tool_result
    ],
}

# gemini
# https://ai.google.dev/gemini-api/docs/function-calling?example=meeting
function_response_part = types.Part.from_function_response(
    name=function_call.name,
    response={"result": result},
)

# Append function call and result of the function execution to contents
contents.append(response.candidates[0].content) # Append the content from the model's response.
contents.append(types.Content(role="user", parts=[function_response_part])) # Append the function response
```


# Tool

```
tools_anthropic = [
    {
        "name": "count_files",
        "description": "Count the number of files in a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "The path to the directory to count files in.",
                },
            },
            "required": ["directory"],
        },
    }
]

tools_google = [x.copy() for x in tools_openai]
[x.pop("type") for x in tools_google]
```

# Batch


(parallem) ~/llmlib$ ~/llmlib/.venv/bin/python ~/llmlib/examples/simplest_batch.py
[DEBUG] Resuming directory
[DEBUG] Creating backend
[DEBUG] Creating provider
[DEBUG] Creating AgentOrchestrator
[INFO] Resuming with session_id=0
Submit 1 batch (1 calls)? (y/n/preview): y
Sent batch: a919pfuvvd9iq9bsq0z4nnwxam0b9hau9zwu
[DASH] ⇈ a919pfuv

(parallem) ~/llmlib$ ~/llmlib/.venv/bin/python ~/llmlib/examples/simplest_batch.py
[DEBUG] Resuming directory
[DEBUG] Creating backend
[DEBUG] Creating provider
[DEBUG] Creating AgentOrchestrator
[INFO] Resuming with session_id=1
Batch a919pfuvvd9iq9bsq0z4nnwxam0b9hau9zwu is still pending.
Cannot proceed until all batches are complete.

(parallem) ~/llmlib$ ~/llmlib/.venv/bin/python ~/llmlib/examples/simplest_batch.py
[DEBUG] Resuming directory
[DEBUG] Creating backend
[DEBUG] Creating provider
[DEBUG] Creating AgentOrchestrator
[INFO] Resuming with session_id=3
Batch a919pfuvvd9iq9bsq0z4nnwxam0b9hau9zwu completed and stored.
Nine (which is $3^2$).
[DASH] C 5efbcfdc


