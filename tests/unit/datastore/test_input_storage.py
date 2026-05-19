import tempfile
from pathlib import Path

import polars as pl
from PIL import Image
from polars.testing import assert_frame_equal

from parallem.core.datastore.input_storage import InputStorage
from parallem.core.file_manager import FileManager
from parallem.types import (
    FunctionCall,
    FunctionCallOutput,
    FunctionCallRequest,
    LLMIdentity,
)


def _sample_call_id(session_id: int, seq_id: int):
    return {
        "agent_name": "test_agent",
        "doc_hash": "test_hash",
        "seq_id": seq_id,
        "session_id": session_id,
        "meta": {"provider_type": "openai", "tag": None},
    }


def _store_input(storage: InputStorage, call_id: dict, msgs: list, instructions=None):
    storage.store_input(
        call_id,
        params={
            "instructions": instructions,
            "llm": LLMIdentity("gpt-4o-mini"),
            "strict_documents": msgs,
            "structured_output": None,
            "tools": None,
        },
        hash_by=None,
        salt=None,
        request_kwargs={},
    )


def test_store_input_text_and_config():
    with tempfile.TemporaryDirectory() as temp_dir:
        fm = FileManager(Path(temp_dir))
        storage = InputStorage(fm)

        call_id = _sample_call_id(session_id=1, seq_id=2)
        _store_input(storage, call_id, ["Hello", "World"], instructions="Test")
        storage.persist()

        text_path = storage.path_inputs_text_table()
        df_text = pl.read_parquet(text_path)
        assert df_text.height == 2
        assert df_text["text"].to_list() == ["Hello", "World"]

        config_parquet = storage.path_inputs_config_parquet(1)
        assert config_parquet.exists()
        items = pl.read_parquet(config_parquet)
        assert items.height == 1
        expected = pl.DataFrame({
            "instructions": ["Test"],
            "llm_identity": "gpt-4o-mini",
        })
        assert_frame_equal(items.select("instructions", "llm_identity"), expected)


def test_store_input_role_tuple():
    with tempfile.TemporaryDirectory() as temp_dir:
        fm = FileManager(Path(temp_dir))
        storage = InputStorage(fm)

        call_id = _sample_call_id(session_id=2, seq_id=1)
        _store_input(storage, call_id, [("user", "Hi")])
        storage.persist()

        text_path = storage.path_inputs_text_table()
        df_text = pl.read_parquet(text_path)
        assert df_text.height == 1
        row = df_text.row(0, named=True)
        assert row["role"] == "user"
        assert row["text"] == "Hi"


def test_store_input_image():
    with tempfile.TemporaryDirectory() as temp_dir:
        fm = FileManager(Path(temp_dir))
        storage = InputStorage(fm)

        call_id = _sample_call_id(session_id=3, seq_id=1)
        img = Image.new("RGB", (5, 5), color="blue")
        _store_input(storage, call_id, [img])
        storage.persist()

        image_index = storage.path_inputs_image_index_table()
        df_images = pl.read_parquet(image_index)
        assert df_images.height == 1
        row = df_images.row(0, named=True)
        img_path = fm.path_inputs() / row["image_path"]
        assert img_path.exists()
        assert img_path.stat().st_size > 0


def test_store_input_function_calls():
    with tempfile.TemporaryDirectory() as temp_dir:
        fm = FileManager(Path(temp_dir))
        storage = InputStorage(fm)

        call_id = _sample_call_id(session_id=4, seq_id=1)
        function_call = FunctionCall(
            name="test_function", arguments={"arg": "value"}, call_id="call_123"
        )
        func_call_req = FunctionCallRequest(
            text_content="Calling function",
            calls=[function_call],
            call_id=call_id,
        )
        func_call_out = FunctionCallOutput(
            content="Function result", call_id="call_123", name="test_function"
        )

        _store_input(storage, call_id, [func_call_req, func_call_out])
        storage.persist()

        req_path = storage.path_inputs_function_call_request_table()
        df_req = pl.read_parquet(req_path)
        assert df_req.height == 1
        assert df_req.row(0, named=True)["text_content"] == "Calling function"

        out_path = storage.path_inputs_function_call_output_table()
        df_out = pl.read_parquet(out_path)
        assert df_out.height == 1
        assert df_out.row(0, named=True)["name"] == "test_function"


def test_store_input_json():
    with tempfile.TemporaryDirectory() as temp_dir:
        fm = FileManager(Path(temp_dir))
        storage = InputStorage(fm)

        call_id = _sample_call_id(session_id=5, seq_id=1)
        _store_input(storage, call_id, [{"a": 1}])
        storage.persist()

        json_path = storage.path_inputs_json_table()
        df_json = pl.read_parquet(json_path)
        assert df_json.height == 1
        assert df_json.row(0, named=True)["json_text"] == "{\"a\":1}"
