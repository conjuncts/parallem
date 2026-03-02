import os
from typing import TYPE_CHECKING, Optional
from parallellm.core.file_manager import FileManager
from pathlib import Path

import sqlite3

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

if TYPE_CHECKING:
    from parallellm.core.datastore.sqlite import SQLiteDatastore


# Helper function to check if a table exists
def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    )
    return cursor.fetchone() is not None


def _check_and_migrate(ds: "SQLiteDatastore") -> None:
    """
    Check if old directory-based structure exists and migrate if needed.
    Only runs once per datastore directory.
    """
    # datastore_dir = ds.file_manager.allocate_datastore()


def _migrate_sql_schema(conn: sqlite3.Connection, db_name: Optional[str]) -> None:
    """
    Migrate SQL schema for a database.

    :param conn: SQLite connection to migrate
    :param db_name: Name of the database (None for main database, or custom names for additional databases)
    """
    try:
        # Add tag column to metadata table if it doesn't exist
        cursor = conn.execute("PRAGMA table_info(metadata)")
        columns = [row[1] for row in cursor.fetchall()]
        if "tag" not in columns:
            conn.execute("ALTER TABLE metadata ADD COLUMN tag TEXT")

        # Add tag column to batch_pending table if it doesn't exist
        if table_exists(conn, "batch_pending"):
            cursor = conn.execute("PRAGMA table_info(batch_pending)")
            columns = [row[1] for row in cursor.fetchall()]
            if "tag" not in columns:
                conn.execute("ALTER TABLE batch_pending ADD COLUMN tag TEXT")

    except sqlite3.Error as e:
        # If migration fails, continue - tables will be created fresh
        db_label = "main database" if db_name is None else f"{db_name} database"
        print(f"Warning: Schema migration failed for {db_label}: {e}")
        return False
    return True


def _remove_unique_constraint(
    conn: sqlite3.Connection, table_name: str, unique_columns: list[str]
) -> None:
    """
    Remove UNIQUE constraint from a specified table by recreating it without the constraint.

    :param conn: SQLite connection
    :param table_name: Name of the table to modify
    :param unique_columns: List of column names that were part of the UNIQUE constraint
    """
    try:
        # Check if table has UNIQUE constraint by examining the schema
        cursor = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
        )
        row = cursor.fetchone()

        if not row or "UNIQUE" not in row[0]:
            return  # No UNIQUE constraint found, nothing to do

        # Get current table structure
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()

        # Build column definitions without UNIQUE constraint
        column_defs = []
        for col in columns_info:
            col_name, col_type, not_null, default_val, pk = (
                col[1],
                col[2],
                col[3],
                col[4],
                col[5],
            )
            col_def = f"{col_name} {col_type}"
            if pk:
                col_def += " PRIMARY KEY"
                if "AUTOINCREMENT" in row[0]:
                    col_def += " AUTOINCREMENT"
            elif not_null:
                col_def += " NOT NULL"
            if default_val is not None:
                col_def += f" DEFAULT {default_val}"
            column_defs.append(col_def)

        conn.execute("BEGIN TRANSACTION")

        # Create new table without UNIQUE constraint
        new_table_name = f"{table_name}_new"
        create_sql = f"CREATE TABLE {new_table_name} ({', '.join(column_defs)})"
        conn.execute(create_sql)

        # Copy data from old table to new table
        column_names = [col[1] for col in columns_info]
        columns_str = ", ".join(column_names)
        conn.execute(
            f"INSERT INTO {new_table_name} SELECT {columns_str} FROM {table_name}"
        )

        # Drop old table and rename new table
        conn.execute(f"DROP TABLE {table_name}")
        conn.execute(f"ALTER TABLE {new_table_name} RENAME TO {table_name}")

        # Recreate indexes (excluding the UNIQUE constraint)
        cursor = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=?",
            (table_name,),
        )
        for index_row in cursor.fetchall():
            if index_row[0] and "UNIQUE" not in index_row[0]:
                # Recreate non-unique indexes
                conn.execute(
                    index_row[0].replace(f"CREATE INDEX", "CREATE INDEX IF NOT EXISTS")
                )

        conn.execute("COMMIT")

    except sqlite3.Error as e:
        # Rollback on error
        try:
            conn.execute("ROLLBACK")
        except:
            pass
        raise RuntimeError(f"Failed to remove UNIQUE constraint from {table_name}: {e}")


def _drop_column(conn: sqlite3.Connection, table_name: str, column: str) -> None:
    """
    Drop a column from a specified table by recreating the table without the column.

    :param conn: SQLite connection
    :param table_name: Name of the table to modify
    :param column: Name of the column to drop
    """
    try:
        # Check if table exists
        if not table_exists(conn, table_name):
            return  # Table doesn't exist, nothing to do

        # Get current table structure
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()

        # Check if column exists
        column_names = [col[1] for col in columns_info]
        if column not in column_names:
            return  # Column doesn't exist, nothing to do

        # Filter out the column to be dropped
        remaining_columns = [col for col in columns_info if col[1] != column]

        # Build column definitions for remaining columns
        column_defs = []
        remaining_names = []
        for col in remaining_columns:
            col_name, col_type, not_null, default_val, pk = (
                col[1],
                col[2],
                col[3],
                col[4],
                col[5],
            )
            col_def = f"{col_name} {col_type}"
            if pk:
                col_def += " PRIMARY KEY"
                # Check if original table had AUTOINCREMENT
                cursor = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,),
                )
                table_sql = cursor.fetchone()
                if table_sql and "AUTOINCREMENT" in table_sql[0]:
                    col_def += " AUTOINCREMENT"
            elif not_null:
                col_def += " NOT NULL"
            if default_val is not None:
                col_def += f" DEFAULT {default_val}"
            column_defs.append(col_def)
            remaining_names.append(col_name)

        conn.execute("BEGIN TRANSACTION")

        # Create new table without the dropped column
        temp_table_name = f"{table_name}_temp"
        create_sql = f"CREATE TABLE {temp_table_name} ({', '.join(column_defs)})"
        conn.execute(create_sql)

        # Copy data from old table to new table (excluding dropped column)
        columns_str = ", ".join(remaining_names)
        conn.execute(
            f"INSERT INTO {temp_table_name} SELECT {columns_str} FROM {table_name}"
        )

        # Drop old table and rename new table
        conn.execute(f"DROP TABLE {table_name}")
        conn.execute(f"ALTER TABLE {temp_table_name} RENAME TO {table_name}")

        # Recreate indexes (excluding any that referenced the dropped column)
        cursor = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=?",
            (table_name,),
        )
        for index_row in cursor.fetchall():
            if index_row[0] and column not in index_row[0]:
                # Only recreate indexes that don't reference the dropped column
                conn.execute(
                    index_row[0].replace(f"CREATE INDEX", "CREATE INDEX IF NOT EXISTS")
                )

        conn.execute("COMMIT")

    except sqlite3.Error as e:
        # Rollback on error
        try:
            conn.execute("ROLLBACK")
        except:
            pass
        raise RuntimeError(f"Failed to drop column {column} from {table_name}: {e}")


def migrate_all_databases_in_directory(root_dir: str) -> None:
    """
    Utility to recursively migrate all datastore.db files in a given root directory.

    :param root_dir: Root directory to recursively search for datastore.db files
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Directory {root_dir} does not exist")
        return

    migrated_count = 0
    failed_count = 0

    # Recursively walk through all directories looking for datastore.db files
    for db_path in root_path.rglob("datastore.db"):
        print(f"Migrating database: {db_path}")
        try:
            # Connect to the database and run migrations
            with sqlite3.connect(str(db_path)) as conn:
                # Use relative path from root as db_name for logging
                relative_path = db_path.relative_to(root_path)
                outcome = _migrate_sql_schema(conn, str(relative_path.parent))
                conn.commit()
            if outcome:
                print(f"  ✓ Successfully migrated {db_path}")
                migrated_count += 1
            else:
                print(f"  ✗ Failed to migrate {db_path}: {e}")
                failed_count += 1
        except Exception as e:
            print(f"  ✗ Failed to migrate {db_path}: {e}")
            failed_count += 1

    print(f"\nMigration complete: {migrated_count} succeeded, {failed_count} failed")


if __name__ == "__main__":
    # Utility to migrate all databases in a directory
    # migrate_all_databases_in_directory(".pllm")
    migrate_all_databases_in_directory("tests/data")
