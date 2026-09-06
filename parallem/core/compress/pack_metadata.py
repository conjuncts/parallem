from pathlib import Path
import gzip

from filelock import FileLock

from parallem.core.compress.to_parquet import ParquetWriter


def compress_metadata_to_zip(metadata_rows: list[str], folder: Path, master_index: ParquetWriter):
    # metadata_rows is actually a sqlite3.Row object

    # Extract metadata strings for processing
    provider_to_meta = {
        "openai": [],
        "google": [],
        "anthropic": [],
        "openai-chat": [],
    }

    for row in metadata_rows:
        row_id = row["id"]
        response_id = row["response_id"]
        metadata_txt = row["metadata"]
        provider_type = row["provider_type"]
        if provider_type not in provider_to_meta:
            provider_to_meta[provider_type] = []

        agent_name = row["agent_name"]
        seq_id = row["seq_id"]
        session_id = row["session_id"]

        master_index.log(
            {
                "response_id": response_id,
                "agent_name": agent_name,
                "seq_id": seq_id,
                "session_id": session_id,
                "provider_type": provider_type,
                "tag": row["tag"],
            }
        )

        if metadata_txt:
            provider_to_meta[provider_type].append((row_id, response_id, metadata_txt))

    if not provider_to_meta:
        return

    # write to .tsv.gz
    stored_row_ids = []
    for provider_type, meta_list in provider_to_meta.items():
        if not meta_list:
            continue

        folder.mkdir(parents=True, exist_ok=True)
        tsv_path = folder / f"{provider_type}-metadata.tsv.gz"
        lock_path = tsv_path.with_suffix(tsv_path.suffix + ".lock")
        with FileLock(lock_path):
            with gzip.open(tsv_path, "at", encoding="utf-8", newline="") as tsv_file:
                for row_id, response_id, metadata_txt in meta_list:
                    if metadata_txt is None:
                        continue
                    tsv_file.write(f"{response_id}\t{metadata_txt}\n")
                    stored_row_ids.append(row_id)
    return stored_row_ids