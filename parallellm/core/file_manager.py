import os
import json
import pickle
import atexit
import re
import hashlib
from pathlib import Path
from typing import Optional

from parallellm.core.state.msg_state import MessageState
from parallellm.types import WorkingMetadata


class FileManager:
    """

    Because FileManager is the first to read metadata, it also is the authoritative
    source for the session_counter / session_id.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata_file = self.directory / "metadata.json"
        self.lock_file = self.directory / ".filemanager.lock"

        # Create directory if it doesn't exist
        self.directory.mkdir(parents=True, exist_ok=True)

        # Create lock file
        self._create_lock()

        # Register cleanup on exit
        atexit.register(self._cleanup)

        # Load existing metadata
        self.metadata = self._load_metadata()
        if self.metadata is None:
            self.metadata = {}
        self.metadata.setdefault(
            "agents",
            {"": {}},
        )

        # session_counter: default to 0
        self.metadata["session_counter"] = self.metadata.get("session_counter", -1) + 1

        # write metadata to immediately persist session_counter increment
        self._save_metadata(self.metadata)

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

    def _load_metadata(self) -> Optional[WorkingMetadata]:
        """Load metadata from JSON file"""
        try:
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _save_metadata(self, metadata: WorkingMetadata):
        """Save metadata to JSON file"""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f)

    def load_agent_msg_state(self, agent_name: str) -> MessageState:
        msg_state_dir = self.directory / "agents" / self._sanitize(agent_name)
        msg_state_file = msg_state_dir / "msg_state.pkl"

        if not msg_state_file.exists():
            return MessageState(agent_name=agent_name)

        with open(msg_state_file, "rb") as f:
            return pickle.load(f)

    def save_agent_msg_state(self, agent_name: str, msg_state: MessageState):
        msg_state_dir = self.directory / "agents" / self._sanitize(agent_name)
        tmp_file = msg_state_dir / "msg_state.tmp.pkl"

        if not tmp_file.parent.exists():
            tmp_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(tmp_file, "wb") as f:
                pickle.dump(msg_state, f)
            # Atomic rename
            os.replace(tmp_file, msg_state_dir / "msg_state.pkl")
        except Exception as e:
            print(f"Failed to save agent message state: {e}")
            raise e
        finally:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)

    def save_userdata(self, key: str, value, overwrite=True):
        """
        Internally persist data

        :param key: The data key
        :param value: The data value to save
        :param overwrite: Whether to overwrite existing data
        """

        # Create userdata directory
        userdata_dir = self.directory / "userdata"
        userdata_dir.mkdir(exist_ok=True)

        # Save data using pickle for complex objects
        fname = self._sanitize(key)
        data_file = userdata_dir / f"{fname}.pkl"

        if data_file.exists() and not overwrite:
            return

        with open(data_file, "wb") as f:
            pickle.dump(value, f)

    def load_userdata(self, key: str):
        """
        Internally load data

        :param key: The data key to load
        :returns: The loaded data
        :raises FileNotFoundError: If the data file is not found
        """
        userdata_dir = self.directory / "userdata"

        fname = self._sanitize(key)
        data_file = userdata_dir / f"{fname}.pkl"

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        with open(data_file, "rb") as f:
            return pickle.load(f)

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
        Write all pending metadata changes to disk
        """
        if self.metadata is not None:
            self._save_metadata(self.metadata)

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
