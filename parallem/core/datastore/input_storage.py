from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

from PIL import Image
import polars as pl

from parallem.core.compress.to_parquet import ParquetWriter, write_to_parquet
from parallem.core.file_manager import FileManager
from parallem.types import (
    CallIdentifier,
    CommonQueryParameters,
    FileInput,
    FunctionCall,
    LLMDocument,
    LLMResponse,
    FunctionCallOutput,
    FunctionCallRequest,
    MCPOutput,
    MultipartDocument,
    to_serial_id,
    LLMIdentity,
    HashByOption,
    ServerTool,
)
from parallem.types import InputStorageConfig
from parallem.utils.image import is_image

if TYPE_CHECKING:
    import pydantic


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_str(text: str) -> str:
    return _hash_bytes(text.encode("utf-8", errors="replace"))


def _parse_fco_content(content_text: Optional[str], content_type: Optional[str]):
    """Reverse the FunctionCallOutput content serialization done in _store_input_doc."""
    if content_text is None:
        return None
    if content_type == "int":
        return int(content_text)
    if content_type == "float":
        return float(content_text)
    if content_type == "bool":
        return content_text == "True"
    if content_type == "json":
        return json.loads(content_text)
    return content_text


class InputStorage:
    """Handle multimedia and request config input storage."""

    def __init__(self, file_manager: FileManager, config: Optional[InputStorageConfig] = None):
        self.input_cfg = config or InputStorageConfig()
        self._file_manager = file_manager
        self._config_log: list[dict] = []

        # Tracks item_hashes already written to each content table this session,
        # so we skip duplicate content writes across calls.
        self._seen_item_hashes: dict[str, set[str]] = {
            "text": set(),
            "json": set(),
            "function_call_request": set(),
            "function_call_output": set(),
            "llm_response": set(),
            "image_inline": set(),
            "file_input": set(),
            "multipart_document": set(),
        }

        # Central index: maps (doc_hash, index_in_msg_state) -> item_hash
        self._item_index_table = ParquetWriter(
            self.path_inputs_multimedia() / "item_index.parquet",
            schema={
                "doc_hash": pl.Utf8,
                "index_in_msg_state": pl.Int64,
                "item_hash": pl.Utf8,
                "item_type": pl.Utf8,
            },
        )

        # Content tables keyed on item_hash (not doc_hash + index)
        self._text_table = ParquetWriter(
            self.path_inputs_text_table(),
            schema={
                "item_hash": pl.Utf8,
                "text": pl.Utf8,
                "role": pl.Utf8,
            },
        )

        self._json_table = ParquetWriter(
            self.path_inputs_json_table(),
            schema={
                "item_hash": pl.Utf8,
                "json_text": pl.Utf8,
            },
        )

        self._function_call_request_table = ParquetWriter(
            self.path_inputs_function_call_request_table(),
            schema={
                "item_hash": pl.Utf8,
                "fcall_id": pl.Utf8,
                "text_content": pl.Utf8,
                "calls_json": pl.Utf8,
            },
        )

        self._function_call_output_table = ParquetWriter(
            self.path_inputs_function_call_output_table(),
            schema={
                "item_hash": pl.Utf8,
                "name": pl.Utf8,
                "fcall_id": pl.Utf8,
                "content_text": pl.Utf8,
                "content_type": pl.Utf8,
            },
        )

        self._llm_response_table = ParquetWriter(
            self.path_inputs_llm_response_table(),
            schema={
                "item_hash": pl.Utf8,
                "call_id": pl.Utf8,
            },
        )

        self._image_inline_table = ParquetWriter(
            self.path_inputs_image_inline_table(),
            schema={
                "item_hash": pl.Utf8,
                "image_path": pl.Utf8,
                "image_format": pl.Utf8,
                "image_bytes": pl.Binary,
            },
        )

        self._file_input_table = ParquetWriter(
            self.path_inputs_multimedia() / "file_inputs.parquet",
            schema={
                "item_hash": pl.Utf8,
                "filename": pl.Utf8,
                "mime_type": pl.Utf8,
                "file_url": pl.Utf8,
                "file_content": pl.Binary,
            },
        )

        self._multipart_document_table = ParquetWriter(
            self.path_inputs_multipart_document_table(),
            schema={
                "item_hash": pl.Utf8,
                "role": pl.Utf8,
                "parts_info": pl.List(pl.Struct({
                    "item_hash": pl.Utf8,
                    "item_type": pl.Utf8,
                })),
                # "parts_json": pl.Utf8,
            },
        )

        self._msg_state_len_table = ParquetWriter(
            self.path_inputs_msg_state_len_table(),
            schema={
                "doc_hash": pl.Utf8,
                "msg_state_len": pl.Int64,
            },
        )

        self.item_tables = {
            # "item_index": self._item_index_table,
            "text": self._text_table,
            "json": self._json_table,
            "function_call_request": self._function_call_request_table,
            "function_call_output": self._function_call_output_table,
            "llm_response": self._llm_response_table,
            "image_inline": self._image_inline_table,
            "file_input": self._file_input_table,
            "multipart_document": self._multipart_document_table,
            # "msg_state_len": self._msg_state_len_table,
        }

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def path_inputs(self) -> Path:
        return self._file_manager.path_inputs()

    def path_inputs_multimedia(self) -> Path:
        multimedia_dir = self.path_inputs() / "media"
        multimedia_dir.mkdir(parents=True, exist_ok=True)
        return multimedia_dir

    def path_inputs_config(self) -> Path:
        config_dir = self.path_inputs() / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def path_inputs_text_table(self) -> Path:
        return self.path_inputs_multimedia() / "text.parquet"

    def path_inputs_json_table(self) -> Path:
        return self.path_inputs_multimedia() / "json.parquet"

    def path_inputs_function_call_request_table(self) -> Path:
        return self.path_inputs_multimedia() / "function_call_request.parquet"

    def path_inputs_function_call_output_table(self) -> Path:
        return self.path_inputs_multimedia() / "function_call_output.parquet"

    def path_inputs_llm_response_table(self) -> Path:
        return self.path_inputs_multimedia() / "llm_response.parquet"

    def path_inputs_image_inline_table(self) -> Path:
        return self.path_inputs_multimedia() / "images_inline.parquet"

    def path_inputs_multipart_document_table(self) -> Path:
        return self.path_inputs_multimedia() / "multipart_document.parquet"

    def path_inputs_msg_state_len_table(self) -> Path:
        return self.path_inputs_multimedia() / "msg_state_len.parquet"

    def path_inputs_image_dir(self) -> Path:
        img_dir = self.path_inputs_multimedia() / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        return img_dir

    def path_inputs_config_zip(self, session_id: int) -> Path:
        return self.path_inputs_config() / f"session_{session_id}.zip"

    def path_inputs_config_parquet(self, session_id: int) -> Path:
        return self.path_inputs_config() / f"session_{session_id}.parquet"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sanitize_for_json(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): self._sanitize_for_json(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize_for_json(item) for item in value]
        return str(value)

    def _image_extension(self, img) -> str:
        fmt = img.format or "PNG"
        fmt = fmt.lower()
        if fmt == "jpeg":
            return "jpg"
        return fmt

    def _log_request_config(
        self,
        call_id: CallIdentifier,
        instructions: Optional[str],
        request_config: dict,
    ) -> None:
        meta = call_id.get("meta", {}) or {}
        record = {
            "session_id": call_id["session_id"],
            "seq_id": call_id["seq_id"],
            "agent_name": call_id["agent_name"],
            "instructions": instructions,
            "provider_type": meta.get("provider_type"),
            "tag": meta.get("tag"),
            "config": self._sanitize_for_json(request_config),
        }
        self._config_log.append(record)

    def _format_structured_output(self, structured_output: "pydantic.BaseModel") -> Optional[str]:
        if structured_output is None:
            return None
        if getattr(structured_output, "model_json_schema", None):
            sch = structured_output.model_json_schema()
            return json.dumps(sch)
        return structured_output.__class__.__name__

    def _format_tools_for_config(
        self,
        tools: Optional[list[Union[dict, ServerTool, Callable]]],
    ) -> Optional[list[dict]]:
        if tools is None:
            return None

        formatted = []
        for tool in tools:
            if isinstance(tool, dict):
                formatted.append(tool)
                continue
            if isinstance(tool, ServerTool):
                formatted.append(
                    {
                        "server_tool_type": tool.server_tool_type,
                        "kwargs": tool.kwargs,
                    }
                )
                continue
            formatted.append({"tool": str(tool)})
        return formatted

    def _build_request_config(
        self,
        *,
        llm: LLMIdentity,
        structured_output,
        tools: Optional[list[Union[dict, ServerTool, Callable]]],
        hash_by: List[HashByOption],
        salt: Optional[str],
        request_kwargs: dict,
    ) -> dict:
        return {
            "llm": {
                "identity": llm.identity,
                "provider_type": llm.provider_type,
                "model_name": llm.model_name,
                "nickname": llm.nickname,
            },
            "structured_output": self._format_structured_output(structured_output),
            "tools": self._format_tools_for_config(tools),
            "hash_by": hash_by,
            "salt": salt,
            "kwargs": request_kwargs,
        }

    def _maybe_write(self, table_name: str, item_hash: str, row: dict) -> None:
        """Write to a content table only if this item_hash hasn't been seen yet."""
        writer = self.item_tables.get(table_name)
        if writer is None:
            return
        if item_hash not in self._seen_item_hashes[table_name]:
            self._seen_item_hashes[table_name].add(item_hash)
            writer.log({
                "item_hash": item_hash,
                **row
            })

    # ------------------------------------------------------------------
    # Core storage logic
    # ------------------------------------------------------------------

    def _store_part(self, part: LLMDocument) -> tuple[str, str]:
        if isinstance(part, tuple):
            role, text = part
            item_hash = _hash_str(f"text\x00{role}\x00{text}")
            self._maybe_write("text", item_hash, {
                "text": text,
                "role": role,
            })
            return item_hash, "text"

        if isinstance(part, str):
            item_hash = _hash_str(f"text\x00\x00{part}")
            self._maybe_write("text", item_hash, {
                "text": part,
                "role": None,
            })
            return item_hash, "text"

        if is_image(part):

            # Still need a stable hash; derive it without saving the file.
            buffered = BytesIO()
            part.save(buffered, format=part.format or "PNG")
            img_bytes = buffered.getvalue()
            item_hash = _hash_bytes(b"image\x00" + img_bytes)
            img_ext = self._image_extension(part)

            if self.input_cfg.save_images_as == "bytes":
                rel_path = None
            elif self.input_cfg.save_images_as == "file":
                img_path = (
                    self.path_inputs_image_dir()
                    / f"{item_hash}.{img_ext}"
                )

                if not img_path.exists():
                    img_path.write_bytes(img_bytes)
                rel_path = img_path.relative_to(self.path_inputs()).as_posix()
                img_bytes = None
            else:
                rel_path = None
                img_bytes = None

            self._maybe_write("image_inline", item_hash, {
                "image_path": rel_path,
                "image_format": img_ext,
                "image_bytes": img_bytes,
            })
            return item_hash, "image_inline"

        if part is None or isinstance(part, (int, float, bool, dict, list)):
            json_text = None
            if self.input_cfg.save_json:
                json_text = json.dumps(part, separators=(",", ":"))
                if (
                    self.input_cfg.json_char_limit is not None
                    and len(json_text) > self.input_cfg.json_char_limit
                ):
                    json_text = None
            item_hash = _hash_str(f"json\x00{json_text}")
            self._maybe_write("json", item_hash, {
                "json_text": json_text,
            })
            return item_hash, "json"

        if isinstance(part, FileInput):
            raw = (part.file_content or b"") + (part.file_url or "").encode()
            item_hash = _hash_bytes(raw + (part.filename or "").encode())
            self._maybe_write("file_input", item_hash, {
                "filename": part.filename,
                "mime_type": part.mime_type,
                "file_url": part.file_url,
                "file_content": part.file_content if self.input_cfg.save_file_contents else None,
            })
            return item_hash, "file_input"

        raise NotImplementedError(f"Unknown document part type: {type(part)}")

    def _store_input_doc(
        self,
        doc_hash: str,
        index_in_msg_state: int,
        msg: Union[LLMDocument, LLMResponse],
    ) -> None:
        if isinstance(msg, MultipartDocument):
            parts_info = []
            for part in msg.parts:
                part_hash, part_type = self._store_part(part)
                parts_info.append({"item_hash": part_hash, "item_type": part_type})

            parts_json = json.dumps(parts_info, separators=(",", ":"))
            item_hash = _hash_str(f"multipart\x00{parts_json}")
            self._item_index_table.log({
                "doc_hash": doc_hash,
                "index_in_msg_state": index_in_msg_state,
                "item_hash": item_hash,
                "item_type": "multipart_document"
            })
            self._maybe_write("multipart_document", item_hash, {
                "parts_info": parts_info,
            })
            return

        if isinstance(msg, (FunctionCallOutput, MCPOutput)):
            name = None
            fcall_id = None
            content_text = None
            content_type = None
            if self.input_cfg.save_function_call_outputs:
                name = msg.name
                fcall_id = msg.fcall_id
                content = msg.content
                if isinstance(content, (str, int, float, bool)):
                    content_text = str(content)
                    content_type = type(content).__name__
                elif isinstance(content, (dict, list)):
                    try:
                        content_text = json.dumps(content, separators=(",", ":"))
                        content_type = "json"
                    except json.JSONDecodeError:
                        content_text = str(content)
                        content_type = "unknown"
                else:
                    content_text = str(content)
                    content_type = "unknown"

            item_hash = _hash_str(
                f"fco\x00{name}\x00{fcall_id}\x00{content_text}\x00{content_type}"
            )
            self._item_index_table.log(
                {"doc_hash": doc_hash, "index_in_msg_state": index_in_msg_state, "item_hash": item_hash, "item_type": "function_call_output"}
            )
            self._maybe_write("function_call_output", item_hash, {
                "name": name,
                "fcall_id": fcall_id,
                "content_text": content_text,
                "content_type": content_type,
            })
            return

        if isinstance(msg, FunctionCallRequest):
            calls = [
                {
                    "name": call.name,
                    "fcall_id": call.fcall_id,
                    "args": self._sanitize_for_json(call.args),
                }
                for call in msg.calls
            ]
            calls_json = json.dumps(calls, separators=(",", ":"))
            call_id = to_serial_id(msg.call_id)
            item_hash = _hash_str(f"fcr\x00{call_id}\x00{msg.text_content}\x00{calls_json}")
            self._item_index_table.log(
                {"doc_hash": doc_hash, "index_in_msg_state": index_in_msg_state, "item_hash": item_hash, "item_type": "function_call_request"}
            )
            self._maybe_write("function_call_request", item_hash, {
                "fcall_id": call_id,
                "text_content": msg.text_content,
                "calls_json": calls_json,
            })
            return

        if isinstance(msg, LLMResponse):
            call_id = to_serial_id(msg.call_id) if msg.call_id is not None else None
            item_hash = _hash_str(f"llmr\x00{call_id}")
            self._item_index_table.log(
                {"doc_hash": doc_hash, "index_in_msg_state": index_in_msg_state, "item_hash": item_hash, "item_type": "llm_response"}
            )
            self._maybe_write("llm_response", item_hash, {
                "call_id": call_id,
            })
            return

        item_hash, item_type = self._store_part(msg)
        self._item_index_table.log(
            {"doc_hash": doc_hash, "index_in_msg_state": index_in_msg_state, "item_hash": item_hash, "item_type": item_type}
        )

    def store_input(
        self,
        call_id: CallIdentifier,
        *,
        params: CommonQueryParameters,
        hash_by: Optional[List[HashByOption]] = None,
        salt: Optional[str] = None,
        request_kwargs: Optional[dict] = None,
    ) -> None:
        instructions = params.get("instructions")
        msgs = params.get("strict_documents")
        llm = params.get("llm")
        structured_output = params.get("structured_output")
        tools = params.get("tools")

        doc_hash = call_id["doc_hash"]
        total_len = len(msgs)
        self._msg_state_len_table.log(
            {
                "doc_hash": doc_hash,
                "msg_state_len": total_len,
            }
        )
        for index, msg in enumerate(msgs):
            self._store_input_doc(doc_hash, index, msg)
        request_config = self._build_request_config(
            llm=llm,
            structured_output=structured_output,
            tools=tools,
            hash_by=hash_by,
            salt=salt,
            request_kwargs=request_kwargs,
        )
        self._log_request_config(call_id, instructions, request_config)

    def retrieve_input(
        self,
        doc_hash: str,
        index_in_msg_state: int,
    ) -> Optional[Union[LLMDocument, LLMResponse]]:
        """
        Retrieve a stored input document by doc_hash and position.

        :param doc_hash: The hash identifying the document batch.
        :param index_in_msg_state: Position of the item in the message state.
        :return: The reconstructed document, or None if not found.
        """

        index_hits = self._item_index_table.get({
            "doc_hash": doc_hash,
            "index_in_msg_state": index_in_msg_state,
        })
        if index_hits.is_empty():
            return None

        item_hash = index_hits["item_hash"][0]
        item_type = index_hits["item_type"][0]

        if item_type not in self.item_tables:
            return None
        part = self._retrieve_part_by_hash(item_hash, item_type)
        if part is not None:
            return part

        rows = self.item_tables[item_type].get({"item_hash": item_hash})
        if rows.is_empty():
            return None
        record = rows.row(0, named=True)
        if item_type == "function_call_request":
            calls_data = json.loads(record["calls_json"] or "[]")
            calls = [
                FunctionCall(name=c["name"], fcall_id=c["fcall_id"], arguments=c["args"])
                for c in calls_data
            ]
            return FunctionCallRequest(
                text_content=record["text_content"],
                calls=calls,
                call_id=None,
            )
        elif item_type == "function_call_output":
            return FunctionCallOutput(
                name=record["name"],
                fcall_id=record["fcall_id"],
                content=_parse_fco_content(record["content_text"], record["content_type"]),
            )
        elif item_type == "llm_response":
            return LLMResponse(value="", call_id=None)
        elif item_type == "multipart_document":
            parts_info: List[Dict[str, str]] = record["parts_info"]
            parts = []
            for p_info in parts_info:
                part_item_hash = p_info["item_hash"]
                part_item_type = p_info["item_type"]
                part_doc = self._retrieve_part_by_hash(part_item_hash, part_item_type)
                if part_doc is not None:
                    parts.append(part_doc)
            return MultipartDocument(parts=parts)

        return None


    def _retrieve_part_by_hash(self, item_hash: str, item_type: str) -> Optional[LLMDocument]:
        if item_type not in {"text", "json", "image_inline", "file_input"}:
            return None
        rows = self.item_tables[item_type].get({"item_hash": item_hash})
        if rows.is_empty():
            return None
        record = rows.row(0, named=True)
        if item_type == "text":
            text = record["text"]
            role = record["role"]
            if role is not None:
                return (role, text)
            return text
        elif item_type == "json":
            json_text = record["json_text"]
            if json_text is not None:
                return json.loads(json_text)
        elif item_type == "image_inline":
            image_path = record["image_path"]
            if image_path is not None:
                full_path = self.path_inputs() / image_path
                if full_path.exists():
                    with Image.open(str(full_path)) as img:
                        img.load()
                        return img
            img_bytes = record["image_bytes"]
            if img_bytes is not None:
                buffered = BytesIO(img_bytes)
                return Image.open(buffered)
        elif item_type == "file_input":
            return FileInput(
                filename=record["filename"],
                mime_type=record["mime_type"],
                file_url=record["file_url"],
                file_content=record["file_content"],
            )
        return None


    def persist(self) -> None:
        # Central index: unique per (doc_hash, index_in_msg_state) pair
        self._item_index_table.commit(mode="unique", on=["doc_hash", "index_in_msg_state"])

        # Content tables: unique per item_hash only
        for _, writer in self.item_tables.items():
            writer.commit(mode="unique", on=["item_hash"])

        self._msg_state_len_table.commit(mode="unique", on=["doc_hash"])

        if not self._config_log:
            return

        session_id = self._config_log[0]["session_id"]
        config_path = self.path_inputs_config_parquet(session_id)
        rows = []
        for record in self._config_log:
            cfg = record.get("config") or {}
            cfg = cfg.copy()

            llm = cfg.pop("llm") or {}
            structured_output = cfg.pop("structured_output")
            tools = cfg.pop("tools")
            hash_by = cfg.pop("hash_by")
            salt = cfg.pop("salt")
            kwargs_obj = cfg.pop("kwargs")

            rows.append(
                {
                    "session_id": record.get("session_id"),
                    "seq_id": record.get("seq_id"),
                    "agent_name": record.get("agent_name"),
                    "instructions": record.get("instructions"),
                    "provider_type": record.get("provider_type"),
                    "tag": record.get("tag"),
                    "llm_identity": llm.get("identity"),
                    "llm_provider_type": llm.get("provider_type"),
                    "llm_model_name": llm.get("model_name"),
                    "llm_nickname": llm.get("nickname"),
                    "structured_output": structured_output,
                    "tools_json": json.dumps(tools, separators=(",", ":"))
                    if tools is not None
                    else None,
                    "hash_by": json.dumps(hash_by, separators=(",", ":"))
                    if hash_by is not None
                    else None,
                    "salt": salt,
                    "kwargs_json": json.dumps(kwargs_obj, separators=(",", ":"))
                    if kwargs_obj is not None
                    else None,
                    "cfg_rest": json.dumps(cfg, separators=(",", ":")),
                }
            )

        write_to_parquet(
            config_path,
            rows,
            mode="append",
            schema={
                "session_id": pl.Int64,
                "seq_id": pl.Int64,
                "agent_name": pl.Utf8,
                "instructions": pl.Utf8,
                "provider_type": pl.Utf8,
                "tag": pl.Utf8,
                "llm_identity": pl.Utf8,
                "llm_provider_type": pl.Utf8,
                "llm_model_name": pl.Utf8,
                "llm_nickname": pl.Utf8,
                "structured_output": pl.Utf8,
                "tools_json": pl.Utf8,
                "hash_by": pl.Utf8,
                "salt": pl.Utf8,
                "kwargs_json": pl.Utf8,
                "cfg_rest": pl.Utf8,
            },
        )

        self._config_log = []
