from itertools import groupby
import sqlite3
import json
import threading
import gzip
import polars as pl
from pathlib import Path
from typing import Literal, Optional

from parallem.core.cast.doc_to_str import cast_bytes_to_document, cast_document_to_bytes
from parallem.core.cast.fix_tools import dump_function_calls, load_function_calls

from parallem.core.datastore.base import BaseDatastore
from parallem.core.datastore.sql_migrate import (
    _check_and_migrate,
    _migrate_sql_schema,
    _ensure_legacy_responses_alias,
    get_schema_version,
    table_exists,
)
from parallem.core.io.sqlite_to_parquet import export_sqlite_to_folder, sqlite_to_df
from parallem.core.compress.pack_metadata import compress_metadata_to_zip
from parallem.core.compress.batch_pending_to_parquet import (
    transfer_batch_pending_to_parquet,
)
from parallem.core.compress.to_parquet import ParquetWriter
from parallem.core.file_manager import FileManager
from parallem.core.memoize.operations import (
    AppendOp,
    ClearOp,
    ExtendOp,
    InsertOp,
    OperationLog,
    PopOp,
    RemoveOp,
    ReverseOp,
    SetNonMsgItemOp,
    SetItemOp,
    SortOp,
)
from parallem.types import (
    BatchIdentifier,
    BatchResult,
    CallIdentifier,
    ParsedError,
    ParsedResponse,
)


class SQLiteDatastore(BaseDatastore):
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

        # Check if migration is needed (only on first initialization)
        self._check_and_migrate()

    def _ensure_memoize_ops_schema(self, conn: sqlite3.Connection) -> None:
        """Ensure memoize_ops contains split item columns.

        :param conn: SQLite connection.
        :return: None.
        """
        cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(memoize_ops)").fetchall()
        }
        if "item_value" not in cols:
            conn.execute("ALTER TABLE memoize_ops ADD COLUMN item_value BLOB")
        if "item_type" not in cols:
            conn.execute("ALTER TABLE memoize_ops ADD COLUMN item_type TEXT")
        if "item_extra" not in cols:
            conn.execute("ALTER TABLE memoize_ops ADD COLUMN item_extra TEXT")
        if "target" not in cols:
            conn.execute(
                "ALTER TABLE memoize_ops ADD COLUMN target TEXT NOT NULL DEFAULT '.msg'"
            )

    def _ensure_responses_schema(self, conn: sqlite3.Connection) -> None:
        """Ensure responses contains origin_type column.

        :param conn: SQLite connection.
        :return: None.
        """
        cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(responses)").fetchall()
        }
        if "origin_type" not in cols:
            conn.execute("ALTER TABLE responses ADD COLUMN origin_type INTEGER")

    def populate_call_id(
        self, short_call_id: dict, *, metadata=False
    ) -> CallIdentifier:
        """Populate a short call_id with doc_hash from the database.

        :param short_call_id: Dict with agent_name, seq_id, session_id
        :return: Full CallIdentifier with doc_hash and meta
        """
        conn = self._get_connection(None)
        cursor = conn.execute(
            """SELECT doc_hash FROM responses 
               WHERE agent_name = ? AND seq_id = ? AND session_id = ?
               ORDER BY id DESC LIMIT 1""",
            (
                short_call_id["agent_name"],
                short_call_id["seq_id"],
                short_call_id["session_id"],
            ),
        )
        row = cursor.fetchone()
        if not row:
            # If not found, return the short call_id
            result = {
                "doc_hash": None,
                "meta": None,
                "agent_name": None,
                "seq_id": None,
                "session_id": None,
            }
            result.update(short_call_id)
            return result

        # Get metadata if available
        meta = None
        if metadata:
            meta_cursor = conn.execute(
                """SELECT provider_type, tag FROM metadata
                WHERE agent_name = ? AND seq_id = ? AND session_id = ?
                LIMIT 1""",
                (
                    short_call_id["agent_name"],
                    short_call_id["seq_id"],
                    short_call_id["session_id"],
                ),
            )
            meta_row = meta_cursor.fetchone()
            if meta_row:
                meta = {
                    "provider_type": meta_row["provider_type"],
                    "tag": meta_row["tag"],
                }

        return {
            "agent_name": short_call_id["agent_name"],
            "doc_hash": row["doc_hash"],
            "seq_id": short_call_id["seq_id"],
            "session_id": short_call_id["session_id"],
            "meta": meta,
        }

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
                self._setup_main_table(conn)
            else:
                raise NotImplementedError(
                    "Only main database connection is implemented"
                )

            conn.commit()
            connections[connection_key] = conn

        self._is_dirty = True
        return connections[connection_key]

    def _setup_main_table(self, conn: sqlite3.Connection) -> None:
        """Create and migrate the main SQLite schema.

        :param conn: SQLite connection.
        :return: None.
        """
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        current_version = get_schema_version(conn)
        has_core_tables = all(
            table_exists(conn, table_name)
            for table_name in [
                "responses",
                "metadata",
                "batch_pending",
                "errors",
                "memoize",
                "memoize_ops",
            ]
        )
        if current_version >= 1 and has_core_tables:
            _ensure_legacy_responses_alias(conn)
            return

        _ensure_legacy_responses_alias(conn)

        # Responses table: agent_name can be NULL
        # No UNIQUE constraint - allows duplicates, retrieve will get most recent (highest id)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                seq_id INTEGER NOT NULL,
                session_id INTEGER NOT NULL,
                doc_hash TEXT NOT NULL,
                response TEXT NOT NULL,
                response_id TEXT,
                tool_calls TEXT,
                origin_type INTEGER
            )
        """)
        self._ensure_responses_schema(conn)

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

        conn.execute("""
            CREATE TABLE IF NOT EXISTS memoize (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                state_hash TEXT NOT NULL,
                final_state TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(agent_name, state_hash)
            )
        """)

        # Structured operation-log rows (safe, no pickle)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memoize_ops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                state_hash TEXT NOT NULL,
                op_seq INTEGER NOT NULL,
                item_seq INTEGER NOT NULL DEFAULT 0,
                op_type TEXT NOT NULL,
                item_value BLOB,
                item_type TEXT,
                item_extra TEXT,
                target TEXT NOT NULL DEFAULT '.msg',
                list_index INTEGER
            )
        """)
        self._ensure_memoize_ops_schema(conn)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memoize_agent_hash
            ON memoize(agent_name, state_hash)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memoize_ops_agent_hash
            ON memoize_ops(agent_name, state_hash)
        """)

        # Migrate existing schema if needed
        _migrate_sql_schema(conn, None)

        # Create indexes for responses table
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_anon_agent_name ON responses(agent_name)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_anon_agent_doc_hash ON responses(agent_name, doc_hash)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_anon_doc_hash ON responses(doc_hash)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_anon_session_id ON responses(session_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_anon_seq_id ON responses(seq_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_anon_response_id ON responses(response_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_anon_origin_type ON responses(origin_type)
        """)

        _ensure_legacy_responses_alias(conn)

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

    def _transfer_metadata_to_parquet(self) -> None:
        """Transfer supported metadata from SQLite to Parquet files."""
        conn = self._get_connection(None)

        try:
            cursor = conn.execute("""
                SELECT m.id, m.response_id, m.agent_name, m.seq_id, m.session_id, m.metadata, m.provider_type, m.tag
                FROM metadata m
            """)
            metadata_rows = cursor.fetchall()

            if not metadata_rows:
                return

            mdir = self.file_manager.path_metadata_store()
            compressed = compress_metadata_to_zip(
                metadata_rows,
                mdir,
                self._metadata_index,
            )
            if compressed:
                self._metadata_index.commit(mode="append")
                # Batch deletes to avoid SQLite's "too many SQL variables" limit (default 999)
                # Process in batches of 500 to stay well under the limit
                batch_size = 500
                for i in range(0, len(compressed), batch_size):
                    batch = compressed[i : i + batch_size]
                    placeholders = ",".join(["?" for _ in batch])
                    conn.execute(
                        f"DELETE FROM metadata WHERE id IN ({placeholders})",
                        batch,
                    )
                conn.commit()

                conn.execute("VACUUM")
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

    def _fetch_oldest_row(
        self,
        conn: sqlite3.Connection,
        table: str,
        where: str,
        params: list,
        *,
        origin_type: Optional[int] = None,
    ) -> Optional[sqlite3.Row]:
        """
        Fetch the oldest (lowest id) row matching a WHERE clause.

        :param conn: SQLite connection.
        :param table: Table name to query.
        :param where: SQL WHERE clause fragment (no leading WHERE keyword).
        :param params: Positional parameters for the WHERE clause.
        :returns: The matching row, or None.
        """
        where_with_origin = f"{where} AND origin_type IS NULL"
        params_with_origin = list(params)
        if origin_type is not None:
            where_with_origin = f"{where} AND origin_type = ?"
            params_with_origin.append(origin_type)

        cursor = conn.execute(
            f"SELECT response, seq_id, session_id, tool_calls"
            f" FROM {table} WHERE {where_with_origin} ORDER BY id ASC LIMIT 1",
            params_with_origin,
        )
        return cursor.fetchone()

    def _parse_tool_calls_from_row(self, row: sqlite3.Row) -> Optional[list]:
        """
        Parse tool_calls JSON from a row, returning None on failure.

        :param row: A SQLite row that contains a ``tool_calls`` column.
        :returns: Parsed function-call list, or None.
        """
        raw = row["tool_calls"]
        if not raw:
            return None
        try:
            return load_function_calls(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    def _row_to_parsed_response(
        self,
        row: sqlite3.Row,
        agent_name: str,
        *,
        include_metadata: bool = False,
    ) -> ParsedResponse:
        """
        Convert a raw SQLite row to a :class:`ParsedResponse`.

        :param row: Row containing ``response``, ``seq_id``, ``session_id``, and ``tool_calls``.
        :param agent_name: Agent name used to look up metadata.
        :param include_metadata: When True, attach retrieved metadata.
        :returns: Populated :class:`ParsedResponse`.
        """
        old_seq_id = row["seq_id"]
        old_session_id = row["session_id"]
        tool_calls = self._parse_tool_calls_from_row(row)
        metadata_value = (
            self.retrieve_metadata(agent_name, old_seq_id, old_session_id)
            if include_metadata
            else None
        )
        return ParsedResponse(
            text=row["response"],
            response_id=None,
            metadata=metadata_value,
            function_calls=tool_calls,
            old_session_id=old_session_id,
            old_seq_id=old_seq_id,
        )

    def retrieve(
        self,
        call_id: CallIdentifier,
        metadata=False,
        *,
        origin_type: Optional[int] = None,
    ) -> Optional[ParsedResponse]:
        """
        Retrieve a response from SQLite.
        Selects the oldest (lowest id) matching entry.

        When ``doc_hash`` is None the lookup is performed solely on
        ``(agent_name, session_id, seq_id)``.  Otherwise the primary lookup
        uses ``(agent_name, doc_hash, seq_id)`` and falls back to
        ``(agent_name, doc_hash)`` if nothing is found.

        :param call_id: The task identifier containing agent_name, doc_hash, seq_id,
            and optionally session_id.
        :param metadata: When True, attach usage metadata to the response.
        :param origin_type: Optional origin marker filter. ``None`` retrieves only
            LLM-originated rows, ``1`` retrieves only human-originated rows.
        :returns: The retrieved response, or None.
        """
        # Oldest entry is chosen to keep retrieval deterministic across concurrent writes.
        doc_hash = call_id["doc_hash"]
        seq_id = call_id["seq_id"]
        agent_name = call_id["agent_name"]

        conn = self._get_connection(None)
        table = "responses"

        if doc_hash is None:
            session_id = call_id.get("session_id")
            row = self._fetch_oldest_row(
                conn,
                table,
                "agent_name = ? AND session_id = ? AND seq_id = ?",
                [agent_name, session_id, seq_id],
                origin_type=origin_type,
            )
        else:
            # Primary: exact doc_hash + seq_id match
            row = self._fetch_oldest_row(
                conn,
                table,
                "agent_name = ? AND doc_hash = ? AND seq_id = ?",
                [agent_name, doc_hash, seq_id],
                origin_type=origin_type,
            )
            if row is None:
                # Fallback: ignore seq_id mismatch
                row = self._fetch_oldest_row(
                    conn,
                    table,
                    "agent_name = ? AND doc_hash = ?",
                    [agent_name, doc_hash],
                    origin_type=origin_type,
                )

        if row is None:
            return None

        return self._row_to_parsed_response(row, agent_name, include_metadata=metadata)

    async def aretrieve(
        self,
        call_id: CallIdentifier,
        metadata=False,
        *,
        origin_type: Optional[int] = None,
    ) -> Optional[ParsedResponse]:
        """
        Asynchronously retrieve a response from SQLite.
        NOTE: not currently async.

        :param call_id: The task identifier containing agent_name, doc_hash, seq_id,
            and optionally session_id.
        :param metadata: When True, attach usage metadata to the response.
        :param origin_type: Optional origin marker filter. ``None`` retrieves only
            LLM-originated rows, ``1`` retrieves only human-originated rows.
        :returns: The retrieved response, or None.
        """
        return self.retrieve(
            call_id,
            metadata=metadata,
            origin_type=origin_type,
        )

    def _read_metadata_tsv(
        self, provider_type: str, response_id: Optional[str]
    ) -> Optional[dict]:
        """
        Read metadata for a response_id from the provider TSV.

        :param provider_type: Provider type used to select the TSV file.
        :param response_id: Response identifier to locate.
        :return: Metadata dictionary or None when not found.
        """
        if not response_id:
            return None

        tsv_path = (
            self.file_manager.path_metadata_store()
            / f"{provider_type}-metadata.tsv.gz"
        )
        if not tsv_path.exists():
            return None

        with gzip.open(tsv_path, "rt", encoding="utf-8", newline="") as tsv_file:
            for line in tsv_file:
                if not line:
                    continue
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                resp_id, metadata_txt = parts
                if resp_id == response_id:
                    try:
                        return json.loads(metadata_txt)
                    except json.JSONDecodeError:
                        return None

        return None

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
        if not self._metadata_index.parquet_fpath.exists():
            return None
        matches = self._metadata_index.get(
            {"agent_name": agent_name, "seq_id": seq_id, "session_id": session_id}
        )
        if matches.height:
            provider_type = matches.item(0, "provider_type")
            resp_id = matches.item(0, "response_id")
            metadata_value = self._read_metadata_tsv(provider_type, resp_id)
            if metadata_value is not None:
                return metadata_value

            parquet_path = (
                self.file_manager.path_metadata_store()
                / f"{provider_type}-responses.parquet"
            )
            if parquet_path.exists():
                relevant = ParquetWriter(parquet_path)
                return relevant.get({"response_id": resp_id}).row(0, named=True)
        return None

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
        origin_type: Optional[int] = None,
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
                "origin_type": origin_type,
            }

            # Insert/upsert the response
            self._insert_response(
                conn,
                "responses",
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
                    "origin_type": None,
                }
                # Insert/upsert the response
                self._insert_response(
                    conn,
                    "responses",
                    record,
                    where_clause="doc_hash = ? AND agent_name = ?" if upsert else None,
                    where_params=[doc_hash, agent_name] if upsert else None,
                    upsert=upsert,
                )

                # Store metadata and tag (tag should always be stored)
                if metadata or tag:
                    metadata_json = json.dumps(metadata) if metadata else ""
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
            transfer_batch_pending_to_parquet(
                conn,
                self.file_manager,
                batch_uuid,
                delete_after_transfer=True,
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

    def export_tables(
        self,
        directory: Optional[str],
        *,
        filetype: Literal["csv", "tsv", "parquet"] = "parquet",
    ) -> None:
        """
        Export all tables from the datastore to files.

        :param directory: Directory to export tables to. If None, uses the default datastore directory.
        :param filetype: Export file type - "polars" or "parquet" for parquet files, "csv" for CSV, "tsv" for TSV.
        """

        # Get the database path
        db_path = self.file_manager.path_datastore() / "datastore.db"

        # Determine export directory
        if directory is None:
            export_dir = self.file_manager.path_datastore() / "export"
        else:
            export_dir = Path(directory)

        # Export the database
        return export_sqlite_to_folder(db_path, export_dir, filetype=filetype)

    def export_polars(
        self,
    ) -> dict[str, pl.DataFrame]:
        """
        Export all tables from the datastore as Polars DataFrames.

        :returns: A dictionary mapping table names to Polars DataFrames.
        """

        # Get the database path
        db_path = self.file_manager.path_datastore() / "datastore.db"

        col = sqlite_to_df(db_path)
        tables = {table_name: df for table_name, df in col}
        return tables

    def import_polars(
        self,
        tables: dict[str, pl.DataFrame],
        *,
        update: bool = True,
    ) -> None:
        """
        Set the datastore state from the provided Polars DataFrames.

        Each key in ``tables`` must match an existing table name.

        When ``update=True`` (default), rows are upserted via ``INSERT OR REPLACE``,
        so existing rows whose primary key matches are replaced in-place while rows
        with new primary keys are simply inserted. The rest of the table is left
        untouched.

        When ``update=False``, existing rows in each named table are deleted before
        inserting the new rows (full overwrite).

        :param tables: A dict mapping table names to Polars DataFrames.
        :param update: If True (default), upsert rows instead of overwriting the table.
        """
        conn = self._get_connection(None)
        normalized_tables: dict[str, pl.DataFrame] = {}
        for table_name, df in tables.items():
            normalized_name = (
                "responses" if table_name == "anon_responses" else table_name
            )
            if normalized_name not in normalized_tables:
                normalized_tables[normalized_name] = df
        try:
            for table_name, df in normalized_tables.items():
                if not update:
                    conn.execute(f"DELETE FROM {table_name}")
                if df.is_empty():
                    continue
                cols = df.columns
                col_names = ", ".join(cols)
                placeholders = ", ".join("?" for _ in cols)
                verb = "INSERT OR REPLACE" if update else "INSERT"
                sql = f"{verb} INTO {table_name} ({col_names}) VALUES ({placeholders})"
                conn.executemany(sql, df.rows())
            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"SQLite error during import: {e}")

    def store_memoize(
        self,
        agent_name: str,
        state_hash: str,
        operation_log: "OperationLog",
    ) -> None:
        """
        Store memoized operation log for a given state hash.

        Each operation is stored as one or more rows in ``memoize_ops``; no pickle
        is used.

        :param agent_name: The name of the agent owning the memoized state.
        :param state_hash: The hash of the initial MessageState.
        :param operation_log: The :class:`~parallem.core.memoize.operations.OperationLog`
            to persist.
        """

        conn = self._get_connection(None)
        self._is_dirty = True

        # Upsert the sentinel row in `memoize` so state_hash is indexed
        conn.execute(
            """
            INSERT OR REPLACE INTO memoize (agent_name, state_hash, timestamp)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (agent_name, state_hash),
        )

        # Remove any previously stored ops for this hash
        conn.execute(
            "DELETE FROM memoize_ops WHERE agent_name = ? AND state_hash = ?",
            (agent_name, state_hash),
        )

        rows: list[tuple] = []
        supported_op_types = {
            "append",
            "remove",
            "clear",
            "reverse",
            "extend",
            "insert",
            "setitem",
            "setnonmsgitem",
            "pop",
            "sort",
        }
        for op_seq, op in enumerate(operation_log.operations):
            if op.op_type not in supported_op_types:
                raise ValueError(f"Unsupported memoize op_type: {op.op_type}")

            target = ".nmsg" if op.op_type == "setnonmsgitem" else ".msg"

            if op.op_type == "setnonmsgitem":
                serialized_items = [cast_document_to_bytes(op.value)]
            elif op.op_type == "extend":
                serialized_items = [cast_document_to_bytes(item) for item in op.items]
            elif op.item is not None:
                serialized_items = [cast_document_to_bytes(op.item)]
            else:
                serialized_items = [(None, None, None)]

            if op.op_type == "sort":
                list_index = 1 if op.reverse else 0
            elif op.op_type in {"insert", "setitem", "pop"}:
                list_index = op.index
            else:
                list_index = None

            for item_seq, (item_value, item_type, item_extra) in enumerate(
                serialized_items
            ):
                extra_payload = item_extra
                if op.op_type == "setnonmsgitem":
                    extra_payload = json.dumps(
                        {
                            "key": op.key,
                            "item_extra": item_extra,
                        },
                        separators=(",", ":"),
                    )

                rows.append(
                    (
                        agent_name,
                        state_hash,
                        op_seq,
                        item_seq,
                        op.op_type,
                        item_value,
                        item_type,
                        extra_payload,
                        target,
                        list_index,
                    )
                )

        conn.executemany(
            """
            INSERT INTO memoize_ops
                (agent_name, state_hash, op_seq, item_seq, op_type, item_value, item_type, item_extra, target, list_index)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def retrieve_memoize(
        self,
        agent_name: str,
        state_hash: str,
    ) -> "Optional[OperationLog]":
        """
        Retrieve memoized operation log for a given state hash.

        :param agent_name: The name of the agent owning the memoized state.
        :param state_hash: The hash of the initial MessageState.
        :return: The reconstructed :class:`~parallem.core.memoize.operations.OperationLog`,
            or ``None`` if not found.
        """

        conn = self._get_connection(None)

        # Check whether any ops exist for this hash
        cursor = conn.execute(
            "SELECT COUNT(*) FROM memoize_ops WHERE agent_name = ? AND state_hash = ?",
            (agent_name, state_hash),
        )
        # count = cursor.fetchone()[0]

        # Also verify the sentinel row exists
        sentinel = conn.execute(
            "SELECT id FROM memoize WHERE agent_name = ? AND state_hash = ?",
            (agent_name, state_hash),
        ).fetchone()
        if sentinel is None:
            return None

        cursor = conn.execute(
            """
            SELECT op_seq, item_seq, op_type, item_value, item_type, item_extra, list_index
            FROM memoize_ops
            WHERE agent_name = ? AND state_hash = ?
            ORDER BY op_seq, item_seq
            """,
            (agent_name, state_hash),
        )
        rows_fetched = cursor.fetchall()

        # Group rows by op_seq
        log = OperationLog()
        for op_seq, group in groupby(rows_fetched, key=lambda r: r["op_seq"]):
            group = list(group)
            op_type = group[0]["op_type"]

            if op_type == "setnonmsgitem":
                extra = (
                    json.loads(group[0]["item_extra"]) if group[0]["item_extra"] else {}
                )
                key = extra.get("key")
                value = cast_bytes_to_document(
                    group[0]["item_value"],
                    group[0]["item_type"],
                    extra.get("item_extra"),
                    retriever=self,
                )
                log.record(SetNonMsgItemOp(key=key, value=value))
                continue

            items = []
            # Hydrate if needed
            for r in group:
                if r["item_value"] is not None:
                    item = cast_bytes_to_document(
                        r["item_value"],
                        r["item_type"],
                        r["item_extra"],
                        retriever=self,
                    )
                    items.append(item)
                else:
                    items.append(None)
            if op_type == "extend":
                log.record(ExtendOp(items))
                continue

            # The other op-types are always one row/op.
            group_0_item = items[0]
            group_0_index = group[0]["list_index"]

            if op_type == "append":
                log.record(AppendOp(group_0_item))
            elif op_type == "insert":
                log.record(
                    InsertOp(
                        index=group_0_index,
                        item=group_0_item,
                    )
                )
            elif op_type == "setitem":
                log.record(
                    SetItemOp(
                        index=group_0_index,
                        item=group_0_item,
                    )
                )
            elif op_type == "pop":
                log.record(PopOp(index=group_0_index))
            elif op_type == "remove":
                log.record(RemoveOp(item=group_0_item))
            elif op_type == "clear":
                log.record(ClearOp())
            elif op_type == "reverse":
                log.record(ReverseOp())
            elif op_type == "sort":
                log.record(SortOp(reverse=bool(group_0_index)))

        return log

    def _vacuum(self):
        """Performs vacuum on the SQLite database."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
