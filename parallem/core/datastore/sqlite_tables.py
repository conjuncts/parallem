import sqlite3
from typing import List, Optional


class SQLiteTable:
    "Encapsulates SQLite table"
    def __init__(
        self,
        name: str,
        creation_sql: str,
        index_sqls: List[str] = None,
    ):
        self.name = name
        self.creation_sql = creation_sql
        self.index_sqls = index_sqls

    def create(self, conn: sqlite3.Connection):
        conn.execute(self.creation_sql)

    def create_indexes(self, conn: sqlite3.Connection):
        if self.index_sqls:
            for index_sql in self.index_sqls:
                conn.execute(f"CREATE INDEX IF NOT EXISTS {index_sql}")

class ResponsesTable(SQLiteTable):
    def __init__(self):
        super().__init__(
            name="responses",
            creation_sql="""
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
            )""",
            index_sqls=[
                "idx_anon_agent_name ON responses(agent_name)",
                "idx_anon_agent_doc_hash ON responses(agent_name, doc_hash)",
                "idx_anon_doc_hash ON responses(doc_hash)",
                "idx_anon_session_id ON responses(session_id)",
                "idx_anon_seq_id ON responses(seq_id)",
                "idx_anon_response_id ON responses(response_id)",
                "idx_anon_origin_type ON responses(origin_type)",
            ]
        )

class MetadataTable(SQLiteTable):
    def __init__(self):
        super().__init__(
            name="metadata",
            creation_sql="""
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
            )""",
            index_sqls=[
                "idx_metadata_response_id ON metadata(response_id)",
                "idx_metadata_provider_type ON metadata(provider_type)",
                "idx_metadata_triple ON metadata(agent_name, seq_id, session_id)",
            ]
        )

    def insert(
        self,
        conn: sqlite3.Connection,
        response_id: str,
        agent_name: str,
        seq_id: int,
        session_id: int,
        metadata_json: str,
        provider_type: Optional[str],
        tag: str,
    ):
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

class BatchPendingTable(SQLiteTable):
    def __init__(self):
        super().__init__(
            name="batch_pending",
            creation_sql="""
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
            )""",
            index_sqls=[
                "idx_batch_pending_batch_uuid ON batch_pending(batch_uuid)",
                "idx_batch_pending_custom_id ON batch_pending(custom_id)",
                "idx_batch_pending_agent_name ON batch_pending(agent_name)",
                "idx_batch_pending_doc_hash ON batch_pending(doc_hash)",
            ]
        )

    def insert(
        self,
        conn: sqlite3.Connection,
        agent_name: str,
        seq_id: int,
        session_id: int,
        doc_hash: str,
        provider_type: Optional[str],
        batch_uuid: str,
        custom_id: str,
        tag: Optional[str],
    ):
        conn.execute(
            """
            INSERT OR REPLACE INTO batch_pending 
            (agent_name, seq_id, session_id, doc_hash, provider_type, batch_uuid, custom_id, tag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_name,
                seq_id,
                session_id,
                doc_hash,
                provider_type,
                batch_uuid,
                custom_id,
                tag,
            ),
        )

class ErrorsTable(SQLiteTable):
    def __init__(self):
        super().__init__(
            name="errors",
            creation_sql="""
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
            )""",
            index_sqls=[
                "idx_errors_agent_name ON errors(agent_name)",
                "idx_errors_doc_hash ON errors(doc_hash)",
                "idx_errors_agent_doc_hash ON errors(agent_name, doc_hash)",
                "idx_errors_session_id ON errors(session_id)",
                "idx_errors_seq_id ON errors(seq_id)",
                "idx_errors_error_code ON errors(error_code)",
            ]
        )

class MemoizeTable(SQLiteTable):
    def __init__(self):
        super().__init__(
            name="memoize",
            creation_sql="""
            CREATE TABLE IF NOT EXISTS memoize (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                state_hash TEXT NOT NULL,
                final_state TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(agent_name, state_hash)
            )""",
            index_sqls=[
                "idx_memoize_agent_hash ON memoize(agent_name, state_hash)",
            ]
        )

class MemoizeOpsTable(SQLiteTable):
    def __init__(self):
        super().__init__(
            name="memoize_ops",
            creation_sql="""
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
            )""",
            index_sqls=[
                "idx_memoize_ops_agent_hash ON memoize_ops(agent_name, state_hash)",
            ]
        )