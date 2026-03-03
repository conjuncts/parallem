import sqlite3
from pathlib import Path
from typing import Generator

import polars as pl


def sqlite_to_df(path_to_db: Path) -> Generator[tuple[str, pl.DataFrame], None, None]:
    """
    Reads a SQLite database and yields each table as a Polars DataFrame.

    :param path_to_db: Path to the SQLite database file.
    :yield: Polars DataFrame for each table in the database.
    """
    conn = sqlite3.connect(str(path_to_db))

    # Get all table names
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    tables = [row[0] for row in cursor.fetchall()]

    # Yield each table as a Polars DataFrame
    for table_name in tables:
        df = pl.read_database(f"SELECT * FROM {table_name}", conn)
        yield table_name, df

    conn.close()


def export_sqlite_to_parquet(path_to_db: Path, folder_for_parquet: Path):
    """
    Turns a SQLite database into parquet files (one per table).

    :param path_to_db: Path to the SQLite database file.
    :param folder_for_parquet: Folder where parquet files will be written.
    """
    folder_for_parquet.mkdir(parents=True, exist_ok=True)

    for table_name, df in sqlite_to_df(path_to_db):
        parquet_file_path = folder_for_parquet / f"{table_name}.parquet"
        df.write_parquet(parquet_file_path)
