import hashlib
import json
from typing import Any, List, Optional, Union

from PIL import Image

from parallem.types import (
    CommonQueryParameters,
    FileInput,
    FunctionCallOutput,
    FunctionCallRequest,
    HashByOption,
    LLMDocument,
    LLMIdentity,
    MCPOutput,
    MultipartDocument,
    ServerTool,
)
from parallem.utils.image import get_type_and_b64


__all__ = ["build_hash_salt_terms", "compute_hash", "serialize_tools_for_hash"]



def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_str(text: str) -> str:
    return _hash_bytes(text.encode("utf-8", errors="replace"))


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
                    schema_str = (
                        json.dumps(
                            structured_output.model_json_schema(),
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if hasattr(structured_output, "model_json_schema")
                        else json.dumps(
                            str(structured_output), sort_keys=True, separators=(",", ":")
                        )
                    )
                    salt_terms.append(schema_str)
            elif term == "kwargs":
                if kwargs:
                    salt_terms.append(
                        json.dumps(kwargs, sort_keys=True, separators=(",", ":"), default=str)
                    )
    return salt_terms


def compute_document_hash(
    doc: LLMDocument,
):
    # TODO: major risk of hash collisions because
    # update(a), update(b) is the same as update(a + b).
    if isinstance(doc, str):
        return _hash_str(f"text\x00\x00{doc}")
    elif isinstance(doc, Image.Image):
        img_type, img_b64 = get_type_and_b64(doc)
        return _hash_str(f"image\x00{img_type}\x00{img_b64}")
    elif isinstance(doc, FunctionCallRequest):
        builder = f"function_call\x00{doc.text_content}"

        for call in doc.calls:
            builder += "\x00" + _hash_str(f"{call.name}\x00{call.arg_str}\x00{call.fcall_id}")
        return _hash_str(builder)
    elif isinstance(doc, FunctionCallOutput):
        return _hash_str(f"function_call_output\x00{doc.name}\x00{str(doc.content or '')}\x00{doc.fcall_id}")
    elif isinstance(doc, MCPOutput):
        return _hash_str(f"mcp_output\x00{doc.name}\x00{str(doc.content or '')}\x00{doc.fcall_id}")
    elif isinstance(doc, FileInput):
        return _hash_str(f"input_file\x00{doc.filename or ''}\x00{doc.file_url or ''}\x00{doc.file_content or ''}")
    elif isinstance(doc, MultipartDocument):
        if len(doc.parts) == 1:
            # if there's only one part, then
            # make sure that MultipartDocument has identical hash to
            # that single part
            if doc.role == "user":
                return compute_document_hash(doc.parts[0])
            else:
                return compute_document_hash((doc.role, doc.parts[0]))
        builder = "multipart"
        for part in doc.parts:
            builder += "\x00" + compute_document_hash(part)
        return _hash_str(builder)
    elif isinstance(doc, tuple) and len(doc) == 2:
        # Handle Tuple[Literal["user", "assistant", "system", "developer"], str]
        role, content = doc
        return _hash_str(f"text\x00{role}\x00{content}")
    elif isinstance(doc, dict):
        # best effort deterministic dict hash
        dict_str = json.dumps(_normalize_for_hash(doc), sort_keys=True, separators=(",", ":"))
        return _hash_str(f"json\x00{dict_str}")
    else:
        raise ValueError(f"Unsupported document type: {type(doc)}")


def compute_hash(
    instructions: Optional[str],
    documents: List[LLMDocument],
    *,
    salt: Optional[str] = None,
) -> str:
    """
    Compute a hash for the given instructions and documents.

    :param instructions: The instructions to hash.
    :param documents: The documents to hash.
    :param salt: An optional salt string. Applied via a second SHA-256 pass over the
        base hash, so it cannot collide with document content.
    :returns: A SHA-256 hash representing the combined content, in hexadecimal format.
    """
    total_hash = ""

    if not instructions:
        instructions = ""
    total_hash += _hash_str(f"system\x00{instructions}")
    for doc in documents:
        total_hash += compute_document_hash(doc)

    base_hash = _hash_str(total_hash)
    if salt is not None:
        # Combine with salt and re-hash to produce final hash
        salted_hasher = hashlib.sha256()
        salted_hasher.update(base_hash.encode("utf-8"))
        salted_hasher.update(str(salt).encode("utf-8"))
        return salted_hasher.hexdigest()
    else:
        return base_hash


def compute_salted_hash(
    params: CommonQueryParameters,
    *,
    salt,
    hash_by,
    kwargs=None,
):
    """Compute the input hash (doc_hash) for a list of documents."""
    # Compute salt
    salt_terms = build_hash_salt_terms(
        salt=salt,
        hash_by=hash_by,
        llm=params["llm"],
        tools=params["tools"],
        structured_output=params["structured_output"],
        kwargs=kwargs,
    )

    # Use a null-byte separator so individual terms cannot be confused with one
    # another, and pass as the `salt` parameter (applied via re-hash) so that
    # salt content can never collide with document content.
    combined_salt = "\x00".join(salt_terms) if salt_terms else None
    hashed = compute_hash(
        params["instructions"],
        params["strict_documents"],
        salt=combined_salt
    )
    return hashed, salt_terms