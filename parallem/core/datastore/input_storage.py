from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import polars as pl

from parallem.core.cast.doc_to_str import cast_document_to_bytes
from parallem.core.compress.pack_zip import persist_to_zip
from parallem.core.compress.to_parquet import ParquetWriter
from parallem.core.file_manager import FileManager
from parallem.types import (
    CallIdentifier,
    LLMDocument,
    LLMResponse,
    FunctionCallOutput,
    FunctionCallRequest,
    to_serial_id,
    LLMIdentity,
    HashByOptions,
    ServerTool,
)
from parallem.provider.openai.openai_tools import to_strict_json_schema
from parallem.utils.image import is_image

if TYPE_CHECKING:
    import pydantic



class InputStorage:
    """Handle multimedia and request config input storage."""

    def __init__(self, file_manager: FileManager):
        self._file_manager = file_manager
        self._config_log: list[dict] = []

        self._text_table = ParquetWriter(
            self.path_inputs_text_table(),
            schema={
                "agent_name": pl.Utf8,
                "session_id": pl.Int64,
                "seq_id": pl.Int64,
                "index_in_msg_state": pl.Int64,
                "text": pl.Utf8,
                "role": pl.Utf8,
            },
        )

        self._json_table = ParquetWriter(
            self.path_inputs_json_table(),
            schema={
                "agent_name": pl.Utf8,
                "session_id": pl.Int64,
                "seq_id": pl.Int64,
                "index_in_msg_state": pl.Int64,
                "json_text": pl.Utf8,
            },
        )

        self._function_call_request_table = ParquetWriter(
            self.path_inputs_function_call_request_table(),
            schema={
                "agent_name": pl.Utf8,
                "session_id": pl.Int64,
                "seq_id": pl.Int64,
                "index_in_msg_state": pl.Int64,
                "call_id": pl.Utf8,
                "text_content": pl.Utf8,
                "calls_json": pl.Utf8,
            },
        )

        self._function_call_output_table = ParquetWriter(
            self.path_inputs_function_call_output_table(),
            schema={
                "agent_name": pl.Utf8,
                "session_id": pl.Int64,
                "seq_id": pl.Int64,
                "index_in_msg_state": pl.Int64,
                "name": pl.Utf8,
                "call_id": pl.Utf8,
                "content_text": pl.Utf8,
            },
        )

        self._llm_response_table = ParquetWriter(
            self.path_inputs_llm_response_table(),
            schema={
                "agent_name": pl.Utf8,
                "session_id": pl.Int64,
                "seq_id": pl.Int64,
                "index_in_msg_state": pl.Int64,
                "call_id": pl.Utf8,
            },
        )

        self._binary_table = ParquetWriter(
            self.path_inputs_binary_table(),
            schema={
                "agent_name": pl.Utf8,
                "session_id": pl.Int64,
                "seq_id": pl.Int64,
                "index_in_msg_state": pl.Int64,
                "doc_type": pl.Utf8,
                "doc_extra": pl.Utf8,
                "doc_value": pl.Binary,
            },
        )

        self._image_index_table = ParquetWriter(
            self.path_inputs_image_index_table(),
            schema={
                "agent_name": pl.Utf8,
                "session_id": pl.Int64,
                "seq_id": pl.Int64,
                "index_in_msg_state": pl.Int64,
                "image_path": pl.Utf8,
                "image_format": pl.Utf8,
            },
        )

        self._msg_state_len_table = ParquetWriter(
            self.path_inputs_msg_state_len_table(),
            schema={
                "agent_name": pl.Utf8,
                "session_id": pl.Int64,
                "seq_id": pl.Int64,
                "msg_state_len": pl.Int64,
            },
        )

    def path_inputs(self) -> Path:
        """
        Get the base inputs directory.

        :returns: Path to the inputs directory.
        """
        return self._file_manager.path_inputs()

    def path_inputs_multimedia(self) -> Path:
        """Get the multimedia inputs directory."""
        multimedia_dir = self.path_inputs() / "multimedia"
        multimedia_dir.mkdir(parents=True, exist_ok=True)
        return multimedia_dir

    def path_inputs_config(self) -> Path:
        """Get the config inputs directory."""
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

    def path_inputs_binary_table(self) -> Path:
        return self.path_inputs_multimedia() / "binary.parquet"

    def path_inputs_image_index_table(self) -> Path:
        return self.path_inputs_multimedia() / "images.parquet"

    def path_inputs_msg_state_len_table(self) -> Path:
        return self.path_inputs_multimedia() / "msg_state_len.parquet"

    def path_inputs_image_dir(self) -> Path:
        img_dir = self.path_inputs_multimedia() / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        return img_dir

    def path_inputs_config_zip(self, session_id: int) -> Path:
        return self.path_inputs_config() / f"session_{session_id}.zip"

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

    def _store_image(self, img) -> tuple[str, str]:
        img_dir = self.path_inputs_image_dir()
        ext = self._image_extension(img)
        buffered = BytesIO()
        img.save(buffered, format=img.format or "PNG")
        img_bytes = buffered.getvalue()
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        img_path = img_dir / f"{img_hash}.{ext}"

        if not img_path.exists():
            img_path.write_bytes(img_bytes)
        rel_path = img_path.relative_to(self.path_inputs()).as_posix()
        return rel_path, ext

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
        return json.dumps(to_strict_json_schema(structured_output))

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
        hash_by: HashByOptions,
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

    def _store_input_doc(
        self,
        agent_name: str,
        session_id: int,
        seq_id: int,
        index_in_msg_state: int,
        msg: Union[LLMDocument, LLMResponse],
    ) -> None:
        if isinstance(msg, tuple):
            role, text = msg
            self._text_table.log(
                {
                    "agent_name": agent_name,
                    "session_id": session_id,
                    "seq_id": seq_id,
                    "index_in_msg_state": index_in_msg_state,
                    "text": text,
                    "role": role,
                }
            )
            return

        if isinstance(msg, str):
            self._text_table.log(
                {
                    "agent_name": agent_name,
                    "session_id": session_id,
                    "seq_id": seq_id,
                    "index_in_msg_state": index_in_msg_state,
                    "text": msg,
                    "role": None,
                }
            )
            return

        if is_image(msg):
            rel_path, img_format = self._store_image(msg)
            self._image_index_table.log(
                {
                    "agent_name": agent_name,
                    "session_id": session_id,
                    "seq_id": seq_id,
                    "index_in_msg_state": index_in_msg_state,
                    "image_path": rel_path,
                    "image_format": img_format,
                }
            )
            return

        if isinstance(msg, FunctionCallOutput):
            self._function_call_output_table.log(
                {
                    "agent_name": agent_name,
                    "session_id": session_id,
                    "seq_id": seq_id,
                    "index_in_msg_state": index_in_msg_state,
                    "name": msg.name,
                    "call_id": msg.call_id,
                    "content_text": str(msg.content),
                }
            )
            return

        if isinstance(msg, FunctionCallRequest):
            calls = [
                {
                    "name": call.name,
                    "call_id": call.call_id,
                    "args": self._sanitize_for_json(call.args),
                }
                for call in msg.calls
            ]
            calls_json = json.dumps(calls, separators=(",", ":"))
            call_id = to_serial_id(msg.call_id)
            self._function_call_request_table.log(
                {
                    "agent_name": agent_name,
                    "session_id": session_id,
                    "seq_id": seq_id,
                    "index_in_msg_state": index_in_msg_state,
                    "call_id": call_id,
                    "text_content": msg.text_content,
                    "calls_json": calls_json,
                }
            )
            return

        if isinstance(msg, LLMResponse):
            call_id = to_serial_id(msg.call_id) if msg.call_id is not None else None
            self._llm_response_table.log(
                {
                    "agent_name": agent_name,
                    "session_id": session_id,
                    "seq_id": seq_id,
                    "index_in_msg_state": index_in_msg_state,
                    "call_id": call_id,
                }
            )
            return

        if msg is None or isinstance(msg, (int, float, bool, dict, list)):
            json_text = json.dumps(msg, separators=(",", ":"))
            self._json_table.log(
                {
                    "agent_name": agent_name,
                    "session_id": session_id,
                    "seq_id": seq_id,
                    "index_in_msg_state": index_in_msg_state,
                    "json_text": json_text,
                }
            )
            return

        content, msg_type, msg_extra = cast_document_to_bytes(msg)
        self._binary_table.log(
            {
                "agent_name": agent_name,
                "session_id": session_id,
                "seq_id": seq_id,
                "index_in_msg_state": index_in_msg_state,
                "doc_type": msg_type,
                "doc_extra": msg_extra,
                "doc_value": content,
            }
        )

    def store_input(
        self,
        call_id: CallIdentifier,
        *,
        instructions: Optional[str],
        msgs: List[Union[LLMDocument, LLMResponse]],
        llm: LLMIdentity,
        structured_output,
        tools: Optional[list[Union[dict, ServerTool, Callable]]],
        hash_by: HashByOptions,
        salt: Optional[str],
        request_kwargs: dict,
    ) -> None:
        session_id = call_id["session_id"]
        seq_id = call_id["seq_id"]
        agent_name = call_id["agent_name"]
        total_len = len(msgs)
        self._msg_state_len_table.log(
            {
                "agent_name": agent_name,
                "session_id": session_id,
                "seq_id": seq_id,
                "msg_state_len": total_len,
            }
        )
        for index, msg in enumerate(msgs):
            self._store_input_doc(agent_name, session_id, seq_id, index, msg)
        request_config = self._build_request_config(
            llm=llm,
            structured_output=structured_output,
            tools=tools,
            hash_by=hash_by,
            salt=salt,
            request_kwargs=request_kwargs,
        )
        self._log_request_config(call_id, instructions, request_config)

    def persist(self) -> None:
        unique_keys = ["agent_name", "session_id", "seq_id", "index_in_msg_state"]
        self._text_table.commit(mode="unique", on=unique_keys)
        self._json_table.commit(mode="unique", on=unique_keys)
        self._function_call_request_table.commit(mode="unique", on=unique_keys)
        self._function_call_output_table.commit(mode="unique", on=unique_keys)
        self._llm_response_table.commit(mode="unique", on=unique_keys)
        self._binary_table.commit(mode="unique", on=unique_keys)
        self._image_index_table.commit(mode="unique", on=unique_keys)
        self._msg_state_len_table.commit(mode="unique", on=["agent_name", "session_id", "seq_id"])

        if not self._config_log:
            return

        grouped: dict[int, list[dict]] = {}
        for record in self._config_log:
            session_id = record["session_id"]
            grouped.setdefault(session_id, []).append(record)

        for session_id, records in grouped.items():
            config_path = self.path_inputs_config_zip(session_id)
            persist_to_zip(
                config_path,
                records,
                inner_fname=f"session_{session_id}",
            )

        self._config_log = []
