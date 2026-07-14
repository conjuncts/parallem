import hashlib
import json
from typing import Any, List, Optional

from PIL import Image

from parallem.core.hash import _normalize_for_hash
from parallem.types import (
    CommonQueryParameters,
    FileInput,
    FunctionCallOutput,
    FunctionCallRequest,
    LLMDocument,
    MCPOutput,
    MultipartDocument,
)
from parallem.utils.image import get_type_and_b64


def _hash_if_present(hasher: "hashlib._Hash", val: Optional[str]):
    if val is not None:
        hasher.update(val.encode("utf-8"))


def old_compute_document_hash(
    hasher: "hashlib._Hash",
    doc: LLMDocument,
):
    # TODO: major risk of hash collisions because
    # update(a), update(b) is the same as update(a + b).
    if isinstance(doc, str):
        hasher.update(doc.encode("utf-8"))
    elif isinstance(doc, Image.Image):
        img_type, img_b64 = get_type_and_b64(doc)
        hasher.update(img_type.encode("utf-8"))
        hasher.update(img_b64.encode("utf-8"))
    elif isinstance(doc, FunctionCallRequest):
        hasher.update(b"function_call")
        _hash_if_present(hasher, doc.text_content)
        for call in doc.calls:
            _hash_if_present(hasher, call.name)
            _hash_if_present(hasher, call.arg_str)
            _hash_if_present(hasher, call.fcall_id)
    elif isinstance(doc, FunctionCallOutput):
        hasher.update(b"function_call_output")
        _hash_if_present(hasher, doc.name)
        _hash_if_present(hasher, str(doc.content))
        _hash_if_present(hasher, doc.fcall_id)
    elif isinstance(doc, MCPOutput):
        hasher.update(b"mcp_output")
        _hash_if_present(hasher, doc.name)
        _hash_if_present(hasher, str(doc.content))
        _hash_if_present(hasher, doc.fcall_id)
    elif isinstance(doc, FileInput):
        hasher.update(b"input_file")
        _hash_if_present(hasher, doc.filename)
        _hash_if_present(hasher, doc.mime_type)
        _hash_if_present(hasher, doc.file_url)
        if doc.file_content is not None:
            hasher.update(doc.file_content)
    elif isinstance(doc, MultipartDocument):
        hasher.update(b"multipart_document")
        for part in doc.parts:
            old_compute_document_hash(hasher, part)
    elif isinstance(doc, tuple) and len(doc) == 2:
        # Handle Tuple[Literal["user", "assistant", "system", "developer"], str]
        role, content = doc
        hasher.update(role.encode("utf-8"))
        if isinstance(content, str):
            hasher.update(content.encode("utf-8"))
        else:
            for item in content:
                hasher.update(str(item).encode("utf-8"))
    elif isinstance(doc, dict):
        # best effort deterministic dict hash
        dict_str = json.dumps(_normalize_for_hash(doc), sort_keys=True, separators=(",", ":"))
        hasher.update(dict_str.encode("utf-8"))
    else:
        raise ValueError(f"Unsupported document type: {type(doc)}")


def old_compute_hash(
    instructions: Optional[str],
    documents: List[LLMDocument],
    *,
    salt: Optional[str] = None,
) -> str:
    """
    Reconstruct the original hash algorithm that was used before the
    ``\x00``-delimited format was introduced.

    The old algorithm fed raw content into an incremental SHA-256 hasher
    without any delimiter between fields, which created a collision risk:
    ``update(a); update(b)`` is the same as ``update(a + b)``.

    :param instructions: The instructions / system prompt.
    :param documents: The document list to hash.
    :param salt: An optional salt string, applied via a second SHA-256 pass.
    :returns: A SHA-256 hex digest matching the legacy format.
    """
    hasher = hashlib.sha256()
    if instructions:
        hasher.update(instructions.encode("utf-8"))
    for doc in documents:
        old_compute_document_hash(hasher, doc)
    base_hash = hasher.hexdigest()
    if salt is not None:
        salted_hasher = hashlib.sha256()
        salted_hasher.update(base_hash.encode("utf-8"))
        salted_hasher.update(str(salt).encode("utf-8"))
        return salted_hasher.hexdigest()
    return base_hash


def old_build_hash_salt_terms(
    *,
    salt: Optional[str] = None,
    hash_by: Optional[List[str]] = None,
    llm: Any,
    provider_type: Optional[str] = None,
    tools: Optional[List[Any]] = None,
    structured_output: Optional[Any] = None,
    kwargs: Optional[dict] = None,
) -> list[str]:
    """
    Build salt terms for the old hash format.

    This mirrors the logic in :func:`parallem.core.hash.build_hash_salt_terms`
    so that the salt used with the old hash is identical to what the new
    code would produce.

    :param salt: Base salt term.
    :param hash_by: Extra hash dimensions.
    :param llm: LLM identity.
    :param provider_type: Provider type (unused in old salt building).
    :param tools: Tool definitions.
    :param structured_output: Structured output schema.
    :param kwargs: Extra kwargs.
    :returns: Ordered salt terms.
    """

    salt_terms: list[str] = []
    if salt is not None:
        salt_terms.append(str(salt))
    if hash_by is not None:
        hash_by_expanded = set(hash_by)
        if "all" in hash_by_expanded:
            hash_by_expanded = {"llm", "tool_names", "structured_output", "kwargs"}

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
                        elif hasattr(tool, "server_tool_type"):
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


def old_compute_salted_hash(
    params: CommonQueryParameters,
    *,
    salt: Optional[str],
    hash_by: Optional[List[str]],
    kwargs: Optional[dict] = None,
):
    """
    Compute the old-format hash (doc_hash) for a set of documents, including
    salt terms.

    :param params: Common query parameters (instructions, documents, llm, …).
    :param salt: Base salt.
    :param hash_by: Extra hash dimensions.
    :param kwargs: Extra request kwargs.
    :returns: Tuple of (old_hex_hash, salt_terms).
    """
    salt_terms = old_build_hash_salt_terms(
        salt=salt,
        hash_by=hash_by,
        llm=params["llm"],
        tools=params.get("tools"),
        structured_output=params.get("structured_output"),
        kwargs=kwargs,
    )
    combined_salt = "\x00".join(salt_terms) if salt_terms else None
    hashed = old_compute_hash(
        params.get("instructions"),
        params.get("strict_documents", []),
        salt=combined_salt,
    )
    return hashed, salt_terms
    