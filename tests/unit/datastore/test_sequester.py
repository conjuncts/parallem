import pytest
import polars as pl
from pathlib import Path
import shutil
from parallellm.core.datastore.sqlite import SQLiteDatastore
from parallellm.core.file_manager import FileManager


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
    # Currently, only openai/google are sequestered
    assert new_count == 3

    # Check that parquet files were created
    metadata_dir = fm.path_metadata_store()
    assert metadata_dir.exists()

    # Check for OpenAI metadata files
    n_openai = list(metadata_dir.glob("openai-*.parquet"))
    assert len(n_openai) == 2

    n_google = list(metadata_dir.glob("google-*.parquet"))
    assert len(n_google) == 1

    # Verify parquet files are readable
    for parquet_file in n_openai:
        df = pl.read_parquet(parquet_file)
        assert not df.is_empty()

    # Verify that metadata can be retrieved
    recovered_meta = ds.retrieve_metadata_legacy("msg_011czEnUFgzYsUye9ygZDinQ")
    assert recovered_meta["model"] == "claude-3-haiku-20240307"

    recovered_meta = ds.retrieve_metadata_legacy(
        "resp_0413f7f758e604110069212d3d1ef0819283556872ee053df7"
    )
    assert recovered_meta["model"] == "gpt-4.1-nano-2025-04-14"

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
