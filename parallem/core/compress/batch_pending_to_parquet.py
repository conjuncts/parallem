from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import polars as pl

from parallem.core.compress.to_parquet import write_to_parquet
from parallem.core.file_manager import FileManager


_BATCH_PENDING_SCHEMA = {
    "id": pl.Int64,
    "agent_name": pl.Utf8,
    "seq_id": pl.Int64,
    "session_id": pl.Int64,
    "doc_hash": pl.Utf8,
    "provider_type": pl.Utf8,
    "batch_uuid": pl.Utf8,
    "custom_id": pl.Utf8,
    "is_pending": pl.Boolean,
    "tag": pl.Utf8,
}


def transfer_batch_pending_to_parquet(
    conn: sqlite3.Connection,
    file_manager: FileManager,
    batch_uuid: str,
    *,
    parquet_fpath: Optional[Path] = None,
    delete_after_transfer: bool = False,
    is_pending: Optional[bool] = None,
) -> int:
    """
    Transfer batch_pending rows to parquet for archival.

    :param conn: SQLite connection.
    :param file_manager: FileManager for locating the datastore directory.
    :param batch_uuid: Batch UUID to archive.
    :param parquet_fpath: Optional explicit parquet file path.
    :param delete_after_transfer: Whether to delete the archived rows after writing.
    :param is_pending: Optional pending filter for archived rows.
    :return: Number of rows archived.
    """
    where = ["batch_uuid = ?"]
    params = [batch_uuid]
    if is_pending is not None:
        where.append("is_pending = ?")
        params.append(1 if is_pending else 0)

    where_sql = " AND ".join(where)
    cursor = conn.execute(
        f"""
		SELECT id, agent_name, seq_id, session_id, doc_hash, provider_type,
		       batch_uuid, custom_id, is_pending, tag
		FROM batch_pending
		WHERE {where_sql}
		ORDER BY id
		""",
        params,
    )
    rows = cursor.fetchall()
    if not rows:
        return 0

    if parquet_fpath is None:
        parquet_fpath = file_manager.path_datastore() / "batch_pending.parquet"

    data = [
        {
            "id": row["id"],
            "agent_name": row["agent_name"],
            "seq_id": row["seq_id"],
            "session_id": row["session_id"],
            "doc_hash": row["doc_hash"],
            "provider_type": row["provider_type"],
            "batch_uuid": row["batch_uuid"],
            "custom_id": row["custom_id"],
            "is_pending": bool(row["is_pending"]),
            "tag": row["tag"],
        }
        for row in rows
    ]

    write_to_parquet(
        parquet_fpath,
        data,
        mode="append",
        schema=_BATCH_PENDING_SCHEMA,
    )

    if delete_after_transfer:
        conn.execute(
            f"DELETE FROM batch_pending WHERE {where_sql}",
            params,
        )

    return len(rows)
