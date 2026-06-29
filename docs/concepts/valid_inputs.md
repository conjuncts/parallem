These can be passed to `ask_llm`:



- LLMDocument
    - `str`
    - `Tuple[Literal["user", "assistant", "system", "developer"], str]`
    - `PIL.Image.Image`
    - `pllm.FunctionCallRequest`
    - `pllm.FunctionCallOutput`
    - `pllm.MCPOutput`
    - `pllm.FileInput`
    - `pllm.ImageURLDocument`
    - `pllm.MultiPartDocument`
- LLMResponse
- The provider's original `dict` format.

## Description of possible inputs


All of the following can be passed into `ask_llm` directly:

1. An **LLMDocument** is an input document for an LLM, which falls under 2 categories.

    1. **Primitive** documents include `str`, `PIL.Image.Image`, and `Tuple[Literal["user", "assistant", "system", "developer"], str]`.

    2. **Non-primitive** documents inherit from **AskItem** , and include `FunctionCallRequest`, `FunctionCallOutput`, `MCPOutput`, `FileInput`, `ImageURLDocument`, and `MultipartDocument`.

1. An **LLMResponse** is an output obtained from an LLM.


## OpenAI Chat Completions API

Documents are required to be 1-to-1 compatible and interconvertible with the [OpenAI Chat Completions API](https://developers.openai.com/api/reference/chat-completions/overview), which has the form:

```
message = {
    "role": "user"
    "content": [
        {
            "type": xxx,
            xxx: yyy
        }
    ]
}
```

| ParaLLeM input type | OpenAI Chat Completions type |
| --- | --- |
| `str` | Message with 1 content part of type `text` |
| `PIL.Image.Image` | Message with 1 content part of type `image_url`, base64-encoded data |
| `Tuple[Literal["user", "assistant", "system", "developer"], str]` | Message with corresponding role |
| `FunctionCallRequest` | Message with `assistant` role and `tool_calls` field |
| `FunctionCallOutput` | Message with `tool` role |
| `MCPOutput` | Message with `tool` role |
| `FileInput` | Message with 1 content part of type `file` |
| `ImageURLDocument` | Message with 1 content part of type `image_url` |
| `MultiPartDocument` | Message with ≥1 content part(s) |

Use `MultipartDocument` to explicitly describe messages which have multiple parts.

### Guarantees

1. While the exact conversion between ParaLLeM and OpenAI Chat Completions may be subject to change, it is guaranteed that valid OpenAI Chat Completions can always be losslessly converted to ParaLLeM's types and back (see `parallem.core.convert.completions.py`)

2. If a `dict` is provided, it will remain untouched.
    - For example, if your provider is OpenAI Responses API, you can pass a `dict` to configure their settings. However, it will not be model agnostic.
