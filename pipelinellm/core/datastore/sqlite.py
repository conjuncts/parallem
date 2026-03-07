import sqlite3
import json
import threading
import polars as pl
from typing import List, Optional, Union

from pipelinellm.core.cast.doc_to_str import cast_document_to_bytes
from pipelinellm.core.cast.fix_tools import dump_function_calls, load_function_calls
from pipelinellm.core.datastore.base import Datastore
from pipelinellm.core.datastore.sql_migrate import (
    _check_and_migrate,
    _migrate_sql_schema,
)
from pipelinellm.core.sink.sequester import sequester_metadata
from pipelinellm.core.sink.to_parquet import ParquetUniqueWriter, ParquetWriter
from pipelinellm.core.file_manager import FileManager
from pipelinellm.types import (
    BatchIdentifier,
    BatchResult,
    CallIdentifier,
    LLMDocument,
    LLMResponse,
    ParsedError,
    ParsedResponse,
)


class SQLiteDatastore(Datastore):
    """
    SQLite-backed Datastore implementation
    """

    def __init__(self, file_manager: FileManager):
        """
        Initialize SQLite Datastore.

        :param file_manager: FileManager instance to handle file I/O operations
        """
        self.file_manager = file_manager
        # Use threading.local to ensure each thread has its own connections
        self._local = threading.local()
        self._is_dirty = False

        self._metadata_index = ParquetWriter(
            self.file_manager.path_metadata_store() / "metadata-index.parquet",
            schema={
                "response_id": pl.Utf8,
                "agent_name": pl.Utf8,
                "seq_id": pl.Int64,
                "session_id": pl.Int64,
                "provider_type": pl.Utf8,
                "tag": pl.Utf8,
            },
        )

        self.history_table = ParquetUniqueWriter(
            self.file_manager.path_history_table(),
            schema={
                "doc_hash": pl.Utf8,
                "instructions": pl.Utf8,
                "msg_hashes": pl.List(pl.Utf8),
                "salt_terms": pl.List(pl.Utf8),
            },
            unique_column_name="doc_hash",
        )

        self.msg_content_table = ParquetUniqueWriter(
            self.file_manager.path_msg_content_table(),
            schema={
                "msg_hash": pl.Utf8,
                "msg_value": pl.Binary,
                "msg_type": pl.Utf8,
                "msg_extra": pl.Utf8,
            },
            unique_column_name="msg_hash",
        )

        # Check if migration is needed (only on first initialization)
        self._check_and_migrate()

    def _check_and_migrate(self) -> None:
        """
        Check if old directory-based structure exists and migrate if needed.
        Only runs once per datastore directory.
        """
        return _check_and_migrate(self)

    def _get_connections(self) -> dict[str, sqlite3.Connection]:
        """Get the connections dictionary for the current thread"""
        if not hasattr(self._local, "connections"):
            self._local.connections = {}
        return self._local.connections

    def _get_connection(self, db_name: Optional[str] = None) -> sqlite3.Connection:
        """
        Get or create SQLite connection for a database.

        :param db_name: Name identifier for the connection (None for main database, or custom names for additional connections)
        :returns: SQLite connection for the specified database
        """
        connections = self._get_connections()

        # Use "main" as key for None db_name
        connection_key = db_name if db_name is not None else "main"

        if connection_key not in connections:
            # Get the base datastore directory
            datastore_dir = self.file_manager.path_datastore()

            # Use main datastore file for None/main, or custom named files for others
            if db_name is None:
                db_path = datastore_dir / "datastore.db"
            else:
                db_path = datastore_dir / f"{db_name}-datastore.db"

            # Create connection
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row  # Enable dict-like access to rows

            # For main database (db_name is None), create response table
            if db_name is None:
                # Responses table: agent_name can be NULL
                # No UNIQUE constraint - allows duplicates, retrieve will get most recent (highest id)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS anon_responses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_name TEXT NOT NULL,
                        seq_id INTEGER NOT NULL,
                        session_id INTEGER NOT NULL,
                        doc_hash TEXT NOT NULL,
                        response TEXT NOT NULL,
                        response_id TEXT,
                        tool_calls TEXT
                    )
                """)

                # Create metadata table (shared between both response tables)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        response_id TEXT,
                        agent_name TEXT NOT NULL,
                        seq_id INTEGER NOT NULL,
                        session_id INTEGER NOT NULL,
                        metadata TEXT NOT NULL,
                        provider_type TEXT,
                        tag TEXT,
                        UNIQUE(response_id)
                    )
                """)

                # Create batch_pending table for storing pending batch requests
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS batch_pending (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_name TEXT NOT NULL,
                        seq_id INTEGER NOT NULL,
                        session_id INTEGER NOT NULL,
                        doc_hash TEXT NOT NULL,
                        provider_type TEXT,
                        batch_uuid TEXT NOT NULL,
                        custom_id TEXT,
                        is_pending BOOLEAN DEFAULT 1,
                        tag TEXT,
                        UNIQUE(custom_id, batch_uuid)
                    )
                """)

                # Create errors table for storing error responses
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS errors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_name TEXT NOT NULL,
                        seq_id INTEGER NOT NULL,
                        session_id INTEGER NOT NULL,
                        doc_hash TEXT NOT NULL,
                        error_message TEXT NOT NULL,
                        error_code INTEGER NOT NULL,
                        error_id TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Migrate existing schema if needed
                _migrate_sql_schema(conn, None)

                # Create indexes for anon_responses table
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_anon_agent_name ON anon_responses(agent_name)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_anon_agent_doc_hash ON anon_responses(agent_name, doc_hash)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_anon_doc_hash ON anon_responses(doc_hash)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_anon_session_id ON anon_responses(session_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_anon_seq_id ON anon_responses(seq_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_anon_response_id ON anon_responses(response_id)
                """)

                # Create indexes for metadata table
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metadata_response_id ON metadata(response_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metadata_provider_type ON metadata(provider_type)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metadata_triple ON metadata(agent_name, seq_id, session_id)
                """)

                # Create indexes for batch_pending table
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_batch_pending_batch_uuid ON batch_pending(batch_uuid)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_batch_pending_custom_id ON batch_pending(custom_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_batch_pending_agent_name ON batch_pending(agent_name)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_batch_pending_doc_hash ON batch_pending(doc_hash)
                """)

                # Create indexes for errors table
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_errors_agent_name ON errors(agent_name)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_errors_doc_hash ON errors(doc_hash)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_errors_agent_doc_hash ON errors(agent_name, doc_hash)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_errors_session_id ON errors(session_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_errors_seq_id ON errors(seq_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_errors_error_code ON errors(error_code)
                """)
            else:
                raise NotImplementedError(
                    "Only main database connection is implemented"
                )

            conn.commit()
            connections[connection_key] = conn

        self._is_dirty = True
        return connections[connection_key]

    def _transfer_metadata_to_parquet(self) -> None:
        """Transfer supported metadata from SQLite to Parquet files."""
        conn = self._get_connection(None)

        try:
            cursor = conn.execute("""
                SELECT m.response_id, m.agent_name, m.seq_id, m.session_id, m.metadata, m.provider_type, m.tag
                FROM metadata m 
                WHERE m.provider_type IN ('openai', 'google') OR m.provider_type IS NULL
            """)
            metadata_rows = cursor.fetchall()

            if not metadata_rows:
                return

            mdir = self.file_manager.path_metadata_store()
            sequestered = sequester_metadata(metadata_rows, mdir, self._metadata_index)
            if sequestered:
                placeholders = ",".join(["?" for _ in sequestered])
                conn.execute(
                    f"DELETE FROM metadata WHERE response_id IN ({placeholders}) AND (provider_type IN ('openai', 'google') OR provider_type IS NULL)",
                    sequestered,
                )
                conn.commit()

                # conn.execute("VACUUM")
        except sqlite3.Error as e:
            raise RuntimeError(f"SQLite error during metadata transfer: {e}")

    def persist(self) -> None:
        """
        Persist (commit) changes to SQLite database and transfer metadata to Parquet files.

        And closes all connections. After this call, the datastore is still usable,
        but current connections are closed (connections will be recreated on demand).

        Also, OpenAI metadata is transferred from SQLite to Parquet files
        for better storage efficiency.
        """
        self.history_table.commit(mode="unique", on="doc_hash")
        self.msg_content_table.commit(mode="unique", on="msg_hash")

        # Transfer all metadata to parquet
        if hasattr(self, "_local") and hasattr(self._local, "connections"):
            try:
                self._transfer_metadata_to_parquet()
                # Refresh parquet manager cache after sequestering
                # self._metadata_parquet.commit()
            except Exception as e:
                # Log the error but don't fail the persist operation
                print(f"Warning: Failed to transfer metadata to Parquet: {e}")

        # Note: SQLite implementation always commits immediately

        # Close all connections to ensure proper cleanup, especially important on Windows
        self.close()

    def close(self, db_name: Optional[str] = None) -> None:
        """
        Close SQLite connection(s) for the current thread.

        :param db_name: The database name to close (if None, close all connections).
        """
        if hasattr(self, "_local") and hasattr(self._local, "connections"):
            connections = self._get_connections()

            if db_name is not None:
                if db_name in connections:
                    connections[db_name].close()
                    del connections[db_name]
            else:
                # Close all connections for current thread
                for conn in connections.values():
                    conn.close()
                connections.clear()

    def __del__(self):
        """
        Cleanup: close all connections when the object is destroyed.
        Note: Only closes connections from the current thread to avoid threading issues.
        """
        try:
            # Only try to close connections if we're in a thread that has them
            if hasattr(self, "_local") and hasattr(self._local, "connections"):
                self.close()
        except Exception:
            # Ignore any errors during cleanup in destructor
            pass

    def retrieve(
        self, call_id: CallIdentifier, metadata=False
    ) -> Optional[ParsedResponse]:
        """
        Retrieve a response from SQLite.
        Selects the oldest entry.

        :param call_id: The task identifier containing agent_name, doc_hash, and seq_id.
        :returns: The retrieved response content.
        """
        # It needs to be the oldest entry, because
        # we want it to be deterministic (we don't want future requests to mess up the order)

        doc_hash = call_id["doc_hash"]
        seq_id = call_id["seq_id"]
        agent_name = call_id["agent_name"]

        conn = self._get_connection(None)
        table_name = "anon_responses"

        where_conditions = "agent_name = ? AND doc_hash = ?"
        params = [agent_name, doc_hash]

        # Ideally, seq_id should match. Get oldest entry
        full_where = where_conditions + " AND seq_id = ?"
        full_params = params + [seq_id]

        cursor = conn.execute(
            f"SELECT response, session_id, tool_calls FROM {table_name} WHERE {full_where} ORDER BY id ASC LIMIT 1",
            full_params,
        )
        old_seq_id = seq_id
        row = cursor.fetchone()
        if not row:
            # Fallback: allow seq_id to differ. Get oldest entry
            cursor = conn.execute(
                f"SELECT response, seq_id, session_id, tool_calls FROM {table_name} WHERE {where_conditions} ORDER BY id ASC LIMIT 1",
                params,
            )
            row = cursor.fetchone()

            if row is None:
                return None

            old_seq_id = row["seq_id"]
        old_session_id = row["session_id"]

        # Parse tool_calls from JSON if present
        tool_calls = None
        if row["tool_calls"]:
            try:
                tool_calls = load_function_calls(row["tool_calls"])
            except (json.JSONDecodeError, TypeError):
                tool_calls = None

        if metadata:
            # Retrieve metadata
            metadata_value = self.retrieve_metadata(
                call_id["agent_name"],
                old_seq_id,
                old_session_id,
            )
        else:
            metadata_value = None

        return ParsedResponse(
            text=row["response"],
            response_id=None,
            metadata=metadata_value,
            function_calls=tool_calls,
            old_session_id=old_session_id,
            old_seq_id=old_seq_id,
        )

    def retrieve_metadata_legacy(self, response_id: str) -> Optional[dict]:
        """
        Retrieve metadata from SQLite and parquet cache using response_id.

        Checks SQLite first, then falls back to cached parquet metadata.

        :param response_id: The response ID to look up metadata for.
        :returns: The retrieved metadata as a dictionary, or None if not found.
        """
        conn = self._get_connection(None)

        cursor = conn.execute(
            "SELECT metadata FROM metadata WHERE response_id = ?",
            (response_id,),
        )
        metadata_row = cursor.fetchone()
        if metadata_row and metadata_row["metadata"]:
            return json.loads(metadata_row["metadata"])

        # If not found in SQLite, check the parquet manager's metadata cache
        # return self._metadata_parquet.get({"response_id": response_id})

        # Guess provider_type (major hack) - but this is legacy anyway
        if response_id.startswith("resp_"):
            provider_type = "openai"
        elif response_id.startswith("msg_"):
            provider_type = "anthropic"
        else:
            provider_type = "google"
        relevant = ParquetWriter(
            self.file_manager.path_metadata_store()
            / f"{provider_type}-responses.parquet"
        )
        return relevant.get({"response_id": response_id}).row(0, named=True)

    def retrieve_metadata(
        self, agent_name: str, seq_id: int, session_id: int
    ) -> Optional[dict]:
        """
        Retrieve metadata from SQLite and parquet cache using agent_name, seq_id, and session_id.

        Checks SQLite first, then falls back to cached parquet metadata.

        :param agent_name: The agent name to look up metadata for.
        :param seq_id: The sequence ID to look up metadata for.
        :param session_id: The session ID to look up metadata for.
        :returns: The retrieved metadata as a dictionary, or None if not found.
        """
        conn = self._get_connection(None)

        cursor = conn.execute(
            "SELECT metadata FROM metadata WHERE agent_name = ? AND seq_id = ? AND session_id = ?",
            (agent_name, seq_id, session_id),
        )
        metadata_row = cursor.fetchone()
        if metadata_row and metadata_row["metadata"]:
            return json.loads(metadata_row["metadata"])

        # If not found in SQLite, check parquet
        matches = self._metadata_index.get(
            {"agent_name": agent_name, "seq_id": seq_id, "session_id": session_id}
        )
        if matches.height:
            provider_type = matches.item(0, "provider_type")
            resp_id = matches.item(0, "response_id")

            relevant = ParquetWriter(
                self.file_manager.path_metadata_store()
                / f"{provider_type}-responses.parquet"
            )
            return relevant.get({"response_id": resp_id}).row(0, named=True)

    def _insert_response(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        record: dict[str, any],
        *,
        where_clause: Optional[str] = None,
        where_params: Optional[list] = None,
        upsert: bool = False,
    ) -> None:
        """
        Insert a response record in the specified table.
        By default always inserts, allowing duplicates.
        If upsert=True, updates the OLDEST existing record (minimum ID) if found.

        :param conn: SQLite connection
        :param table_name: Name of the table to insert into
        :param record: Dictionary mapping column names to values
        :param where_clause: WHERE clause for checking existence (required if upsert=True)
        :param where_params: Parameters for the WHERE clause (required if upsert=True)
        :param upsert: If True, update oldest (min ID) existing record instead of inserting duplicate
        """
        columns = list(record.keys())
        values = list(record.values())

        if upsert:
            if where_clause is None or where_params is None:
                raise ValueError("where_clause and where_params required for upsert")

            # Check if record already exists, get the one with minimum ID (oldest)
            cursor = conn.execute(
                f"SELECT id FROM {table_name} WHERE {where_clause} ORDER BY id ASC LIMIT 1",
                where_params,
            )
            existing = cursor.fetchone()

            if existing:
                # Update the oldest existing record (minimum ID)
                set_clauses = [f"{col} = ?" for col in columns]
                update_sql = (
                    f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE id = ?"
                )
                conn.execute(update_sql, values + [existing["id"]])
                return

        # Insert new record (either upsert with no existing, or normal insert)
        placeholders = ", ".join(["?" for _ in columns])
        insert_sql = (
            f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        )
        conn.execute(insert_sql, values)

    def store(
        self,
        call_id: CallIdentifier,
        parsed_response: "ParsedResponse",
        *,
        upsert: bool = False,
    ) -> None:
        doc_hash = call_id["doc_hash"]
        seq_id = call_id["seq_id"]
        session_id = call_id["session_id"]
        agent_name = call_id["agent_name"]

        call_meta = call_id.get("meta", {})
        provider_type = call_meta.get("provider_type")
        tag = call_meta.get("tag")

        response = parsed_response.text
        response_id = parsed_response.response_id
        metadata = parsed_response.metadata
        tool_calls = parsed_response.function_calls

        conn = self._get_connection(None)

        try:
            tool_calls_json = dump_function_calls(tool_calls)

            record = {
                "agent_name": agent_name,
                "seq_id": seq_id,
                "session_id": session_id,
                "doc_hash": doc_hash,
                "response": response,
                # "response_id": response_id,
                "tool_calls": tool_calls_json,
            }

            # Insert/upsert the response
            self._insert_response(
                conn,
                "anon_responses",
                record,
                where_clause="doc_hash = ? AND agent_name = ?" if upsert else None,
                where_params=[doc_hash, agent_name] if upsert else None,
                upsert=upsert,
            )

            # Store metadata if provided
            if metadata:
                metadata_json = json.dumps(metadata)
                conn.execute(
                    "INSERT OR REPLACE INTO metadata (response_id, agent_name, seq_id, session_id, metadata, provider_type, tag) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        response_id,
                        agent_name,
                        seq_id,
                        session_id,
                        metadata_json,
                        provider_type,
                        tag,
                    ),
                )

            # Always commit immediately for thread safety
            conn.commit()

        except sqlite3.Error as e:
            conn.rollback()
            raise RuntimeError(f"SQLite error while storing response: {e}")

    def store_error(
        self,
        call_id: CallIdentifier,
        err: ParsedError,
    ) -> None:
        """
        Store an error response in the backend.

        :param call_id: The task identifier containing doc_hash, seq_id, and session_id.
        :param err: The error response object containing error details.
        """

        conn = self._get_connection(None)

        try:
            # Insert the error record
            conn.execute(
                """
                INSERT INTO errors (agent_name, seq_id, session_id, doc_hash, error_message, error_code, error_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call_id["agent_name"],
                    call_id["seq_id"],
                    call_id["session_id"],
                    call_id["doc_hash"],
                    err.msg,
                    err.error_code,
                    err.error_id,
                ),
            )

            # Always commit immediately for thread safety
            conn.commit()

        except sqlite3.Error as e:
            conn.rollback()
            raise RuntimeError(f"SQLite error while storing error: {e}")

    def store_input(
        self,
        doc_hash: str,
        *,
        instructions: Optional[str],
        msgs: List[Union[LLMDocument, LLMResponse]],
        msg_hashes: list[str],
        salt_terms: list[str],
    ):
        self.history_table.log_kv(
            doc_hash,
            {
                "instructions": instructions,
                "msg_hashes": msg_hashes,
                "salt_terms": salt_terms,
            },
        )

        if len(msgs) != len(msg_hashes):
            raise ValueError(
                f"Number of documents ({len(msgs)}) must equal number of hashes ({len(msg_hashes)})"
            )

        for msg, msg_hash in zip(msgs, msg_hashes):
            val = cast_document_to_bytes(msg)
            if val is not None:
                content, msg_type, msg_extra = val
                self.msg_content_table.log_kv(
                    msg_hash,
                    {
                        "msg_value": content,
                        "msg_type": msg_type,
                        "msg_extra": msg_extra,
                    },
                )

    def store_pending_batch(
        self,
        batch_id: BatchIdentifier,
    ) -> None:
        conn = self._get_connection(None)

        try:
            # Insert each call_id from the batch into the batch_pending table
            for call_id, custom_id in zip(batch_id.call_ids, batch_id.custom_ids):
                call_meta = call_id.get("meta", {})

                # Insert or replace the pending batch record
                conn.execute(
                    """
                    INSERT OR REPLACE INTO batch_pending 
                    (agent_name, seq_id, session_id, doc_hash, provider_type, batch_uuid, custom_id, tag)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        call_id["agent_name"],
                        call_id["seq_id"],
                        call_id["session_id"],
                        call_id["doc_hash"],
                        call_meta.get("provider_type"),
                        batch_id.batch_uuid,
                        custom_id,
                        call_meta.get("tag"),
                    ),
                )

            # Commit immediately for thread safety
            conn.commit()

        except sqlite3.Error as e:
            conn.rollback()
            raise RuntimeError(f"SQLite error while storing batch: {e}")

    def store_ready_batch(
        self,
        batch_result: BatchResult,
        *,
        upsert: bool = False,
    ) -> None:
        if not batch_result.parsed_responses:
            return

        conn = self._get_connection(None)

        try:
            # Process each response in the batch
            for i, parsed in enumerate(batch_result.parsed_responses):
                custom_id = parsed.custom_id
                # Look up the call_id using custom_id from active batch_pending

                if parsed.error_code != 0:
                    # TODO: This means there is an error. Should be stored separately.
                    pass
                cursor = conn.execute(
                    """
                    SELECT agent_name, seq_id, session_id, doc_hash, provider_type, tag
                    FROM batch_pending
                    WHERE custom_id = ? AND is_pending = 1
                    LIMIT 1
                    """,
                    (custom_id,),
                )
                row = cursor.fetchone()

                if not row:
                    raise ValueError(
                        f"Could not find pending batch record for custom_id: {custom_id}"
                    )

                agent_name = row["agent_name"]
                seq_id = row["seq_id"]
                session_id = row["session_id"]
                doc_hash = row["doc_hash"]
                provider_type = row["provider_type"]
                tag = row["tag"]

                resp_text = parsed.text
                response_id = parsed.response_id
                metadata = parsed.metadata
                tool_calls = parsed.function_calls

                # Serialize tool_calls to JSON if present
                tool_calls_json = dump_function_calls(tool_calls)

                # Prepare record for INSERT/UPDATE
                record = {
                    "agent_name": agent_name,
                    "seq_id": seq_id,
                    "session_id": session_id,
                    "doc_hash": doc_hash,
                    "response": resp_text,
                    # "response_id": custom_id,
                    # NB: response_id column is deprecated
                    "tool_calls": tool_calls_json,
                }
                # Insert/upsert the response
                self._insert_response(
                    conn,
                    "anon_responses",
                    record,
                    where_clause="doc_hash = ? AND agent_name = ?" if upsert else None,
                    where_params=[doc_hash, agent_name] if upsert else None,
                    upsert=upsert,
                )

                # Store metadata if available
                if metadata:
                    metadata_json = json.dumps(metadata)
                    conn.execute(
                        "INSERT OR REPLACE INTO metadata (response_id, agent_name, seq_id, session_id, metadata, provider_type, tag) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            response_id,
                            agent_name,
                            seq_id,
                            session_id,
                            metadata_json,
                            provider_type,
                            tag,
                        ),
                    )

            conn.commit()

        except sqlite3.Error as e:
            conn.rollback()
            raise RuntimeError(f"SQLite error while storing batch results: {e}")

    def retrieve_batch_call_ids(self, batch_uuid: str) -> list[CallIdentifier]:
        conn = self._get_connection(None)

        cursor = conn.execute(
            """
            SELECT agent_name, seq_id, session_id, doc_hash, provider_type
            FROM batch_pending
            WHERE batch_uuid = ? AND is_pending = 1
            ORDER BY seq_id
            """,
            (batch_uuid,),
        )
        rows = cursor.fetchall()

        call_ids = []
        for row in rows:
            call_id: CallIdentifier = {
                "agent_name": row["agent_name"],
                "seq_id": row["seq_id"],
                "session_id": row["session_id"],
                "doc_hash": row["doc_hash"],
                "provider_type": row["provider_type"],
            }
            call_ids.append(call_id)

        return call_ids

    def get_all_pending_batch_uuids(self) -> list[tuple[str, str]]:
        """
        Get tuples of (batch_uuid, provider_type) for all active pending batches.
        """
        conn = self._get_connection(None)

        # Get all unique batch_uuids that are still active
        cursor = conn.execute(
            """
            SELECT DISTINCT batch_uuid, provider_type
            FROM batch_pending
            WHERE is_pending = 1
            ORDER BY batch_uuid
            """
        )
        batch_uuids = [
            (row["batch_uuid"], row["provider_type"]) for row in cursor.fetchall()
        ]

        return batch_uuids

    def clear_batch_pending(self, batch_uuid: str) -> None:
        conn = self._get_connection(None)

        try:
            conn.execute(
                "UPDATE batch_pending SET is_pending = 0 WHERE batch_uuid = ?",
                (batch_uuid,),
            )
            conn.commit()

        except sqlite3.Error as e:
            conn.rollback()
            raise RuntimeError(f"SQLite error while deactivating batch: {e}")

    def is_call_in_pending_batch(self, call_id: CallIdentifier) -> bool:
        conn = self._get_connection(None)

        agent_name = call_id["agent_name"]
        doc_hash = call_id["doc_hash"]

        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM batch_pending "
            "WHERE doc_hash = ? AND agent_name = ? AND is_pending = 1",
            (doc_hash, agent_name),
        )
        row = cursor.fetchone()
        return row["count"] > 0 if row else False
