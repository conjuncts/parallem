from pathlib import Path
from typing import Dict, Optional
import polars as pl

from parallem.core.compress.to_parquet import ParquetWriter, write_to_parquet


def compress_metadata(
    metadata_rows: list[Dict], folder: Path, master_index: ParquetWriter
) -> Optional[list[str]]:
    """
    Compress OpenAI metadata from SQLite rows to Parquet files.
    Returns a list of response_ids that were successfully transferred and can be deleted from SQLite.
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

    response_ids_to_delete = master_index.commit(
        mode="append", receipt_col="response_id"
    )
    if response_ids_to_delete is not None:
        response_ids_to_delete = (
            response_ids_to_delete.select("response_id").to_series().to_list()
        )

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
