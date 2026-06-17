from parallem.core.compress.pack_zip import compress_file_to_zip, read_zip_text


def test_compress_openai_input_batch_file_to_zip_removes_source_and_preserves_content(
    tmp_path,
):
    src_fpath = tmp_path / "batch_input.jsonl"
    payload = '{"custom_id": "req_1"}\n{"custom_id": "req_2"}\n'
    src_fpath.write_bytes(payload.encode("utf-8"))

    zip_fpath = compress_file_to_zip(src_fpath)

    assert zip_fpath.exists()
    assert not src_fpath.exists()

    zipped_payload = read_zip_text(zip_fpath, "batch_input.jsonl")

    assert zipped_payload == payload
