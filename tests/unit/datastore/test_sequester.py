import pytest
import polars as pl
from pathlib import Path
import shutil
import gzip
import json
from parallem.core.datastore.sqlite import SQLiteDatastore
from parallem.core.file_manager import FileManager


@pytest.fixture
def test_wkdir(tmp_path):
    """Create a copy of the test database."""
    source_wkdir = Path("tests/data/compress-test")
    dest_wkdir = tmp_path / "compress-test"
    shutil.copytree(source_wkdir, dest_wkdir)
    return dest_wkdir


def test_sequester_metadata(test_wkdir):
    """Test sequestering metadata from SQLite to Parquet."""
    # Connect to the test database copy
    fm = FileManager(test_wkdir)
    ds = SQLiteDatastore(fm)

    conn = ds._get_connection()
    cursor = conn.cursor()

    # Fetch metadata rows
    cursor.execute("SELECT response_id, metadata, provider_type FROM metadata")
    metadata_rows = cursor.fetchall()

    # Store original count
    original_count = len(metadata_rows)
    assert original_count == 9

    # Run the sequester function
    ds._transfer_metadata_to_parquet()

    # Assertions
    # Check the new count is less than or equal to original
    cursor = conn.cursor()
    cursor.execute("SELECT response_id, metadata, provider_type FROM metadata")
    new_rows = cursor.fetchall()
    new_count = len(new_rows)
    # openai/google/anthropic are sequestered
    assert new_count == 0

    # Check that parquet files were created
    metadata_dir = fm.path_metadata_store()
    assert metadata_dir.exists()

    # Check for OpenAI metadata files (now written as tsv.gz)
    n_openai = list(metadata_dir.glob("openai-metadata.tsv.gz"))
    assert len(n_openai) == 1

    n_google = list(metadata_dir.glob("google-metadata.tsv.gz"))
    assert len(n_google) == 1

    openai_tsv = {}
    with gzip.open(metadata_dir / "openai-metadata.tsv.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            resp_id, metadata_txt = line.split("\t", 1)
            openai_tsv[resp_id] = json.loads(metadata_txt)

    assert len(openai_tsv) == 3
    assert (
        openai_tsv["resp_0413f7f758e604110069212d3d1ef0819283556872ee053df7"]["model"]
        == "gpt-4.1-nano-2025-04-14"
    )

    anthropic_tsv = {}
    with gzip.open(metadata_dir / "anthropic-metadata.tsv.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            resp_id, metadata_txt = line.split("\t", 1)
            anthropic_tsv[resp_id] = json.loads(metadata_txt)
    assert len(anthropic_tsv) == 3
    assert anthropic_tsv["msg_011czEnUFgzYsUye9ygZDinQ"]["model"] == "claude-3-haiku-20240307"
    conn.close()


if __name__ == "__main__":
    # Debug
    wkdir = Path("experiments/debug-compress-test")
    shutil.rmtree(wkdir, ignore_errors=True)
    shutil.copytree(
        Path("tests/data/compress-test"),
        wkdir,
        dirs_exist_ok=True,
    )
    fm = FileManager(wkdir)
    ds = SQLiteDatastore(fm)
    ds._transfer_metadata_to_parquet()
