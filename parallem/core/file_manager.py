import os
import json
import atexit
import re
import hashlib
import sqlite3
from pathlib import Path
from typing import Optional

from parallem.types import WorkingMetadata


class FileManager:
    """

    Because FileManager is the first to read metadata, it also is the authoritative
    source for the session_counter / session_id.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.lock_file = self.directory / ".filemanager.lock"
        self.session_counter_db = self.directory / "session_counter.sqlite3"

        # Create directory if it doesn't exist
        self.directory.mkdir(parents=True, exist_ok=True)

        # Create lock file
        self._create_lock()

        # Register cleanup on exit
        atexit.register(self._cleanup)

        legacy_session_counter = self._load_legacy_session_counter()
        self.metadata: WorkingMetadata = {
            "session_counter": -1,
        }

        # session_counter: atomically allocated via SQLite
        self.metadata["session_counter"] = self._allocate_session_counter(
            seed=legacy_session_counter
        )

        self.batch_group_counter = 0

    def _get_session_counter(self) -> int:
        """Get the current session ID"""
        return self.metadata.get("session_counter", None)

    def _sanitize(
        self, user_input: Optional[str], *, default="default", add_hash=True
    ) -> str:
        """
        Sanitize user input to be safe for use as directory name.
        Uses format: <first_64_chars>-<8_letter_hash>.

        :param user_input: The string to sanitize
        :returns: Sanitized string
        """
        if user_input is None:
            return default

        if not isinstance(user_input, str):
            user_input = str(user_input)

        cleaned = "".join(
            [c if (c.isalnum() or c in " _-") else "_" for c in user_input]
        )

        # Replace multiple spaces/underscores with single ones
        cleaned = re.sub(r"[_\s]+", "_", cleaned)

        # Remove leading/trailing whitespace, underscores, and dots
        cleaned = cleaned.strip().strip("_")

        # Take first 64 characters
        if len(cleaned) > 64:
            cleaned = cleaned[:64].rstrip("_.")

        if not cleaned:
            cleaned = "empty"

        # Return format: chk-<first_64_chars>-<8_letter_hash>
        if not add_hash:
            return cleaned

        input_hash = hashlib.sha256(user_input.encode("utf-8")).hexdigest()[:8]
        return f"{cleaned}-{input_hash}"

    def _create_lock(self):
        """Create lock file with current process ID"""
        with open(self.lock_file, "w") as f:
            f.write(str(os.getpid()))

    def _allocate_session_counter(self, *, seed: int = -1) -> int:
        """
        Atomically increment and return the global session counter.

        :param seed: Minimum baseline for migration from legacy metadata.json counter.
        :return: Newly allocated unique session counter
        """
        conn = sqlite3.connect(
            self.session_counter_db, timeout=30, isolation_level=None
        )
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS counters (
                    name TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO counters(name, value) VALUES ('session_counter', -1)"
            )
            conn.execute(
                """
                UPDATE counters
                SET value = CASE WHEN value < ? THEN ? ELSE value END
                WHERE name = 'session_counter'
                """,
                (seed, seed),
            )
            conn.execute(
                "UPDATE counters SET value = value + 1 WHERE name = 'session_counter'"
            )
            row = conn.execute(
                "SELECT value FROM counters WHERE name = 'session_counter'"
            ).fetchone()
            conn.commit()
            if row is None:
                raise RuntimeError("Failed to allocate session counter")
            return int(row[0])
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _load_legacy_session_counter(self) -> int:
        """
        Read previous session counter from legacy metadata.json, if present.

        :return: Last known legacy counter value, or -1 when unavailable
        """
        legacy_metadata_file = self.directory / "metadata.json"
        try:
            with open(legacy_metadata_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return -1

        legacy_counter = payload.get("session_counter")
        if isinstance(legacy_counter, int):
            return legacy_counter
        return -1

    def _cleanup(self):
        """Cleanup method called on object destruction"""
        if self.lock_file.exists():
            try:
                self.lock_file.unlink()
            except (FileNotFoundError, PermissionError):
                pass  # Lock was already removed or can't be removed

    # def __del__(self):
    #     """Destructor to ensure cleanup"""
    #     self._cleanup()

    def path_datastore(self) -> Path:
        """
        Get the base datastore directory.

        :returns: Path to the datastore directory
        """
        datastore_dir = self.directory / "datastore"
        datastore_dir.mkdir(parents=True, exist_ok=True)
        return datastore_dir

    def path_metadata_store(self) -> Path:
        """
        Get directory for dumping metadata
        """
        folder = self.path_datastore() / "apimeta"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def path_inputs(self) -> Path:
        """
        Get the base inputs directory.

        :returns: Path to the inputs directory
        """
        inputs_dir = self.directory / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        return inputs_dir

    def path_history_table(self) -> Path:
        return self.path_inputs() / "history_table.parquet"

    def path_msg_content_table(self) -> Path:
        return self.path_inputs() / "msg_content_table.parquet"

    def path_batch_in(self) -> Path:
        """
        Get the base batches directory.

        :returns: Path to the batch inputs directory
        """
        batch_dir = self.directory / "batch-in"
        batch_dir.mkdir(parents=True, exist_ok=True)
        return batch_dir

    def save_batch_in(
        self, stuff: list[dict], *, preferred_name=None, batch_counter_id=None
    ):
        """
        Helper function to persist batch inputs to disk.
        Coordinates with the FileManager to get a suitable location.
        `stuff` is JSON-serialized.

        Because many SDKs want the entire batch to be sent over as a file.
        """
        if batch_counter_id is None:
            batch_counter_id = self.batch_group_counter
            self.batch_group_counter += 1

        if preferred_name is None:
            preferred_name = (
                f"batch_{self._get_session_counter()}_{batch_counter_id}.jsonl"
            )

        # Remnants of same-session_id files should not be possible, since
        # session_id should be unique per run.
        path = self.path_batch_in() / preferred_name

        with open(path, "w", encoding="utf-8") as f:
            for item in stuff:
                f.write(json.dumps(item) + "\n")

        return path

    def path_batch_out(self) -> Path:
        """
        Get the base batch outputs directory.

        :returns: Path to the batch outputs directory
        """
        batch_dir = self.directory / "batch-out"
        batch_dir.mkdir(parents=True, exist_ok=True)
        return batch_dir

    def persist(self):
        """
        Persist FileManager state.

        Session counters are persisted in SQLite at allocation time.
        Agent metadata is currently in-memory only.
        """
        return None

    def is_locked(self):
        """
        Check if another FileManager instance has locked this directory
        """
        if not self.lock_file.exists():
            return False

        try:
            with open(self.lock_file, "r") as f:
                lock_pid = int(f.read().strip())
                # Check if the process is still running (Windows-specific)
                try:
                    os.kill(lock_pid, 0)
                    return True  # Process exists
                except OSError:
                    return False  # Process doesn't exist
        except (ValueError, FileNotFoundError):
            return False
