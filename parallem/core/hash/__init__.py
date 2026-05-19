import hashlib
import json
from io import BytesIO
from typing import Any, List, Optional, Union

from PIL import Image
import warnings

from parallem.types import (
    FunctionCallOutput,
    FunctionCallRequest,
    HashByOption,
    LLMDocument,
    LLMIdentity,
    ServerTool,
)
from parallem.utils.image import get_type_and_b64


__all__ = ["build_hash_salt_terms", "compute_hash", "serialize_tools_for_hash"]


def _updateh(hasher, val: Optional[str]):
    if val is not None:
        hasher.update(val.encode("utf-8"))


def _normalize_for_hash(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): _normalize_for_hash(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(item) for item in value]

    if isinstance(value, ServerTool):
        attrs = {}
        if hasattr(value, "__dict__"):
            attrs = {
                str(key): _normalize_for_hash(val)
                for key, val in sorted(value.__dict__.items(), key=lambda item: item[0])
                if not key.startswith("_")
            }
        return {"server_tool_type": value.server_tool_type, "attrs": attrs}

    if callable(value):
        module_name = getattr(value, "__module__", "")
        qualname = getattr(value, "__qualname__", getattr(value, "__name__", ""))
        return f"{module_name}.{qualname}".strip(".")

    return repr(value)


def serialize_tools_for_hash(tools: Optional[list[Union[dict, ServerTool]]]) -> str:
    """
    Serialize tools deterministically for use in a hash salt.

    :param tools: Tools to serialize.
    :return: Deterministic JSON string for the provided tools.
    """
    return json.dumps(_normalize_for_hash(tools), sort_keys=True, separators=(",", ":"))


def build_hash_salt_terms(
    *,
    salt: Optional[str] = None,
    hash_by: List[HashByOption] = None,
    llm: LLMIdentity,
    provider_type: Optional[str] = None,
    tools: Optional[list[Union[dict, ServerTool]]] = None,
    structured_output: Optional[Any] = None,
    kwargs: Optional[dict] = None,
) -> list[str]:
    """
    Build the salt terms used to differentiate cache keys.

    :param salt: Base salt term.
    :param hash_by: Extra hash dimensions to include. Available options:
        - "llm": Include the LLM identity
        - "tool_names": Include tool names only
        - "structured_output": Include structured output schema
        - "kwargs": Include extra kwargs
        - "all": Include everything (llm, tools, tool_names, structured_output, kwargs)
    :param llm: Selected LLM identity, if any.
    :param provider_type: Provider type for the current call.
    :param tools: Tools available to the LLM.
    :param structured_output: Structured output schema, if any.
    :param kwargs: Extra kwargs passed to the request.
    :return: Ordered salt terms to join into the final salt string.
    """
    salt_terms: list[str] = []
    if salt is not None:
        salt_terms.append(str(salt))
    if hash_by is not None:
        # Expand "all" to include all hash options
        hash_by_expanded = set(hash_by)
        if "all" in hash_by_expanded:
            hash_by_expanded = ["llm", "tool_names", "structured_output", "kwargs"]
        
        for term in hash_by_expanded:
            if term == "llm":
                salt_terms.append(llm.identity)
            elif term == "tool_names":
                if tools is not None:
                    tool_names = []
                    for tool in tools:
                        if isinstance(tool, dict):
                            if "name" in tool:
                                tool_names.append(tool["name"])
                            elif "type" in tool:
                                tool_names.append(tool["type"])
                        elif isinstance(tool, ServerTool):
                            tool_names.append(tool.server_tool_type)
                    if tool_names:
                        salt_terms.append(json.dumps(tool_names, sort_keys=True))
            elif term == "structured_output":
                if structured_output is not None:
                    schema_str = json.dumps(structured_output.model_json_schema(), sort_keys=True, separators=(",", ":")) if hasattr(structured_output, "model_json_schema") else json.dumps(str(structured_output), sort_keys=True, separators=(",", ":"))
                    salt_terms.append(schema_str)
            elif term == "kwargs":
                if kwargs:
                    salt_terms.append(json.dumps(kwargs, sort_keys=True, separators=(",", ":"), default=str))
    return salt_terms


def compute_hash(
    instructions: Optional[str],
    documents: List[LLMDocument],
    *,
    salt: Optional[str] = None,
    _legacy_salt: Optional[List[str]] = None,
) -> str:
    """
    Compute a hash for the given instructions and documents.

    :param instructions: The instructions to hash.
    :param documents: The documents to hash.
    :param salt: An optional salt string. Applied via a second SHA-256 pass over the
        base hash, so it cannot collide with document content.
    :param _legacy_salt: Deprecated. For migration only. Reproduces the old
        (collision-prone) behaviour where salt terms were appended directly to the
        document list. Pass the same list that was formerly concatenated onto the
        documents argument to recover legacy hashes.
    :returns: A SHA-256 hash representing the combined content, in hexadecimal format.
    """
    if _legacy_salt is not None:
        warnings.warn(
            "_legacy_salt is deprecated and exists only for migration purposes. "
            "Switch to the `salt` parameter to avoid hash collisions.",
            DeprecationWarning,
            stacklevel=2,
        )
        return compute_hash(instructions, list(documents) + _legacy_salt)
    hasher = hashlib.sha256()
    if instructions:
        hasher.update(instructions.encode("utf-8"))
    for doc in documents:
        if isinstance(doc, str):
            hasher.update(doc.encode("utf-8"))
        elif isinstance(doc, Image.Image):
            img_type, img_b64 = get_type_and_b64(doc)
            hasher.update(img_type.encode("utf-8"))
            hasher.update(img_b64.encode("utf-8"))
        elif isinstance(doc, FunctionCallRequest):
            hasher.update(b"function_call")
            _updateh(hasher, doc.text_content)
            for call in doc.calls:
                # hasher.update(str(call).encode("utf-8"))
                _updateh(hasher, call.name)
                _updateh(hasher, call.arg_str)
                _updateh(hasher, call.call_id)
        elif isinstance(doc, FunctionCallOutput):
            hasher.update(b"function_call_output")
            _updateh(hasher, doc.name)
            _updateh(hasher, str(doc.content))
            _updateh(hasher, doc.call_id)
        elif isinstance(doc, tuple) and len(doc) == 2:
            # Handle Tuple[Literal["user", "assistant", "system", "developer"], str]
            role, content = doc
            hasher.update(role.encode("utf-8"))
            if isinstance(content, str):
                hasher.update(content.encode("utf-8"))
            else:
                for item in content:
                    hasher.update(str(item).encode("utf-8"))
        else:
            raise ValueError(f"Unsupported document type: {type(doc)}")

    base_hash = hasher.hexdigest()
    if salt is not None:
        # Combine with salt and re-hash to produce final hash
        salted_hasher = hashlib.sha256()
        salted_hasher.update(base_hash.encode("utf-8"))
        salted_hasher.update(str(salt).encode("utf-8"))
        return salted_hasher.hexdigest()
    else:
        return base_hash
