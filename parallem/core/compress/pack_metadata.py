from pathlib import Path
from typing import Dict, Optional
import gzip
import polars as pl

from parallem.core.compress.to_parquet import ParquetWriter, write_to_parquet


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
        with gzip.open(tsv_path, "at", encoding="utf-8", newline="") as tsv_file:
            for row_id, response_id, metadata_txt in meta_list:
                if metadata_txt is None:
                    continue
                tsv_file.write(f"{response_id}\t{metadata_txt}\n")
                stored_row_ids.append(row_id)
    return stored_row_ids


def compress_metadata(
    metadata_rows: list[Dict], folder: Path, master_index: ParquetWriter
) -> Optional[list[str]]:
    """
    Compress metadata rows into a zip archive.

    The metadata payload is written as a TSV file without JSON encoding.

    :param metadata_rows: SQLite rows containing metadata columns.
    :param folder: Output folder for zip archives.
    :param master_index: Parquet index writer for lookup metadata.
    :return: Response IDs that can be deleted from SQLite.
    """
    # metadata_rows is actually a sqlite3.Row object

    # Extract metadata strings for processing
    provider_to_meta = {
        "openai": [],
        "google": [],
    }

    for row in metadata_rows:
        response_id = row["response_id"]
        metadata_json = row["metadata"]
        provider_type = row["provider_type"]

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

        if metadata_json and provider_type in provider_to_meta:
            provider_to_meta[provider_type].append(
                (
                    {
                        "response_id": response_id,
                    },
                    metadata_json,
                )
            )

    if not provider_to_meta:
        return

    # Process metadata using the existing sinker function
    _openai_met = provider_to_meta["openai"]
    if _openai_met:
        from parallem.provider.openai._compress import compress_openai_metadata

        processed_dfs = compress_openai_metadata(_openai_met)
        _sequester_dfs(processed_dfs, folder, provider_type="openai")

    _google_met = provider_to_meta["google"]
    if _google_met:
        from parallem.provider.google._compress import compress_google_metadata

        processed_dfs = compress_google_metadata(_google_met)
        _sequester_dfs(processed_dfs, folder, provider_type="google")

    response_ids_to_delete = master_index.commit(mode="append", receipt_col="response_id")
    if response_ids_to_delete is not None:
        response_ids_to_delete = response_ids_to_delete.select("response_id").to_series().to_list()

    return response_ids_to_delete


def _sequester_dfs(dfs: dict[str, pl.DataFrame], folder: Path, provider_type: str):
    folder.mkdir(parents=True, exist_ok=True)

    for df_name, df in dfs.items():
        if df.is_empty():
            continue

        write_to_parquet(
            folder / f"{provider_type}-{df_name}.parquet",
            df,
            mode="append",
        )
