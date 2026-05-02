import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional

from parallem.types import FunctionCall, FunctionCallOutput, LLMDocument, LLMResponse

if TYPE_CHECKING:
    from parallem.core.agent.agent import AgentContext


class _LazyOpenAIResponse:
    """Lazy-loading OpenAI response wrapper that defers LLMResponse resolution."""

    _UNSET = object()

    def __init__(
        self,
        llm_response: LLMResponse,
        model: Optional[str],
        text_format: Optional[Any] = None,
    ):
        self._llm_response = llm_response
        self._model = model
        self._text_format = text_format
        self._resolved_text: Optional[str] = None
        self._resolved_output: Optional[list[dict]] = None
        self._resolved_output_parsed: Any = self._UNSET
        self._is_resolved = False

    def _resolve_once(self) -> None:
        """Resolve the underlying LLMResponse only once."""
        if self._is_resolved:
            return

        self._resolved_text = self._llm_response.final_answer or ""

        self._resolved_output: list[dict] = []
        for call in self._llm_response.resolve_function_calls():
            if isinstance(call, FunctionCall):
                self._resolved_output.append(
                    {
                        "type": "function_call",
                        "name": call.name,
                        "arguments": call.arg_str,
                        "call_id": call.call_id,
                    }
                )

        if self._resolved_text:
            self._resolved_output.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": self._resolved_text}],
                }
            )

        self._is_resolved = True

    def _resolve_output_parsed_once(self) -> None:
        if self._resolved_output_parsed is not self._UNSET:
            return

        self._resolve_once()
        self._resolved_output_parsed = _coerce_output_parsed(
            self._resolved_text,
            self._text_format,
        )

    def __getattr__(self, key: str):
        if key in {"output_text", "output", "output_parsed"}:
            self._resolve_once()

        if key == "output_parsed":
            self._resolve_output_parsed_once()
            return self._resolved_output_parsed

        if key == "output_text":
            return self._resolved_text
        if key == "output":
            return self._resolved_output
        if key == "id":
            pr = getattr(self._llm_response, "_pr", None)
            return pr.response_id if pr is not None else None
        if key == "model":
            return self._model
        if key == "metadata":
            pr = getattr(self._llm_response, "_pr", None)
            return pr.metadata if pr is not None else None

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


class _AwaitableLazyOpenAIResponse:
    """Awaitable wrapper that defers LLMResponse resolution until await time."""

    def __init__(
        self,
        llm_response: LLMResponse,
        model: Optional[str],
        text_format: Optional[Any] = None,
    ):
        self._llm_response = llm_response
        self._model = model
        self._text_format = text_format
        self._payload: Optional[_LazyOpenAIResponse] = None

    async def _resolve_and_return(self) -> _LazyOpenAIResponse:
        """Await the LLMResponse and return the lazy payload."""
        await self._llm_response
        self._payload = _LazyOpenAIResponse(
            self._llm_response,
            self._model,
            self._text_format,
        )
        return self._payload

    def __await__(self):
        return self._resolve_and_return().__await__()

    def _ensure_payload(self) -> _LazyOpenAIResponse:
        if self._payload is None:
            self._payload = _LazyOpenAIResponse(
                self._llm_response,
                self._model,
                self._text_format,
            )
        return self._payload

    def __getattr__(self, key: str):
        return getattr(self._ensure_payload(), key)


def _openai_input_to_documents(payload: Any) -> list[LLMDocument]:
    if payload is None:
        return []

    items = payload if isinstance(payload, list) else [payload]
    docs: list[LLMDocument] = []

    for item in items:
        if isinstance(item, str):
            docs.append(item)
            continue

        if isinstance(item, tuple) and len(item) == 2:
            docs.append(item)
            continue

        if not isinstance(item, dict):
            raise ValueError(f"Unsupported OpenAI input item type: {type(item)}")

        if item.get("type") == "function_call_output":
            docs.append(
                FunctionCallOutput(
                    content=item.get("output"),
                    call_id=item.get("call_id", ""),
                    name=item.get("name", "tool"),
                )
            )
            continue

        role = item.get("role")
        if role is None:
            raise ValueError(
                "OpenAI input messages must include either 'role' or type='function_call_output'."
            )

        content = item.get("content", "")
        if isinstance(content, str):
            docs.append((role, content))
            continue

        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"input_text", "output_text", "text"}:
                    txt = part.get("text")
                    if txt is not None:
                        text_parts.append(str(txt))
            docs.append((role, "".join(text_parts)))
            continue

        docs.append((role, str(content)))

    return docs


def _llm_response_to_openai_payload(
    response: LLMResponse,
    *,
    model: Optional[str],
    text_format: Optional[Any] = None,
) -> _LazyOpenAIResponse:
    return _LazyOpenAIResponse(response, model, text_format)


def _coerce_output_parsed(output_text: str, text_format: Any) -> Any:
    if output_text is None:
        return None

    if text_format is not None and hasattr(text_format, "model_validate_json"):
        try:
            return text_format.model_validate_json(output_text)
        except Exception:
            return None

    try:
        return json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        return None


class _OpenAICompatResponsesSync:
    def __init__(self, agent: "AgentContext"):
        self._agent = agent

    def create(
        self,
        *,
        model: str,
        instructions: Optional[str] = None,
        input: Any = None,
        tools: Optional[list] = None,
        text_format: Optional[object] = None,
        **kwargs,
    ) -> _LazyOpenAIResponse:
        documents = _openai_input_to_documents(input)
        llm_response = self._agent.ask_llm(
            documents,
            instructions=instructions,
            llm=model,
            tools=tools,
            structured_output=text_format,
            **kwargs,
        )
        return _llm_response_to_openai_payload(
            llm_response,
            model=model,
            text_format=text_format,
        )

    def parse(
        self,
        *,
        model: str,
        instructions: Optional[str] = None,
        input: Any = None,
        tools: Optional[list] = None,
        text_format: Optional[object] = None,
        **kwargs,
    ) -> _LazyOpenAIResponse:
        return self.create(
            model=model,
            instructions=instructions,
            input=input,
            tools=tools,
            text_format=text_format,
            **kwargs,
        )


class _OpenAICompatResponsesAsync:
    def __init__(self, agent: "AgentContext"):
        self._agent = agent

    async def create(
        self,
        *,
        model: str,
        instructions: Optional[str] = None,
        input: Any = None,
        tools: Optional[list] = None,
        text_format: Optional[object] = None,
        **kwargs,
    ) -> _AwaitableLazyOpenAIResponse:
        """Return an awaitable response without eagerly awaiting the underlying LLMResponse."""
        documents = _openai_input_to_documents(input)
        llm_response = self._agent.ask_llm(
            documents,
            instructions=instructions,
            llm=model,
            tools=tools,
            structured_output=text_format,
            **kwargs,
        )
        return _AwaitableLazyOpenAIResponse(llm_response, model, text_format)

    async def parse(
        self,
        *,
        model: str,
        instructions: Optional[str] = None,
        input: Any = None,
        tools: Optional[list] = None,
        text_format: Optional[object] = None,
        **kwargs,
    ) -> _AwaitableLazyOpenAIResponse:
        return await self.create(
            model=model,
            instructions=instructions,
            input=input,
            tools=tools,
            text_format=text_format,
            **kwargs,
        )


class _OpenAICompatChatCompletionsSync:
    def __init__(self, responses_ns: _OpenAICompatResponsesSync):
        self._responses_ns = responses_ns

    def create(self, *, model: str, messages: list, **kwargs) -> SimpleNamespace:
        resp = self._responses_ns.create(model=model, input=messages, **kwargs)
        return SimpleNamespace(
            id=resp.id,
            object="chat.completion",
            model=model,
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": resp.output_text,
                    },
                }
            ],
        )


class _OpenAICompatChatCompletionsAsync:
    def __init__(self, responses_ns: _OpenAICompatResponsesAsync):
        self._responses_ns = responses_ns

    async def create(self, *, model: str, messages: list, **kwargs) -> SimpleNamespace:
        resp = await self._responses_ns.create(model=model, input=messages, **kwargs)
        return SimpleNamespace(
            id=resp.id,
            object="chat.completion",
            model=model,
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": resp.output_text,
                    },
                }
            ],
        )


class OpenAICompatClient:
    def __init__(
        self,
        agent: "AgentContext",
        *,
        strategy: Optional[str] = "sync",
    ):
        if strategy == "concurrent":
            responses_ns = _OpenAICompatResponsesAsync(agent)
            completions_ns = _OpenAICompatChatCompletionsAsync(responses_ns)
        else:
            responses_ns = _OpenAICompatResponsesSync(agent)
            completions_ns = _OpenAICompatChatCompletionsSync(responses_ns)

        self.responses = responses_ns
        self.chat = SimpleNamespace(completions=completions_ns)
