from typing import TYPE_CHECKING, Optional
import zipfile

from parallem.core.compress.pack_zip import compress_file_to_zip, persist_to_zip
from parallem.provider.base import BatchProvider
from parallem.types import ProviderType


if TYPE_CHECKING:
    from parallem.core.agent.orchestrator import AgentOrchestrator
    from parallem.core.datastore.sqlite import SQLiteDatastore


def _archive_inactive_batch_pending(
    ds: "SQLiteDatastore",
    *,
    delete_after_transfer: bool = True,
) -> int:
    """
    Archive inactive batch_pending rows to parquet.

    :param delete_after_transfer: Whether to delete archived rows.
    :return: Number of rows archived.
    """
    from parallem.core.compress.batch_pending_to_parquet import (
        transfer_batch_pending_to_parquet,
    )
    import sqlite3

    conn = ds._get_connection(None)
    try:
        cursor = conn.execute(
            """
            SELECT DISTINCT batch_uuid
            FROM batch_pending
            WHERE is_pending = 0
            ORDER BY batch_uuid
            """
        )
        batch_uuids = [row["batch_uuid"] for row in cursor.fetchall()]
        archived = 0
        for batch_uuid in batch_uuids:
            archived += transfer_batch_pending_to_parquet(
                conn,
                ds.file_manager,
                batch_uuid,
                delete_after_transfer=delete_after_transfer,
                is_pending=False,
            )
        conn.commit()
        return archived
    except sqlite3.Error as e:
        conn.rollback()
        raise RuntimeError(f"SQLite error while archiving batch pending: {e}")


class BatchNamespace:
    """Namespace for batch operations."""

    def __init__(self, orch: "AgentOrchestrator"):
        self._orch = orch

    def _compress_inputs(
        self,
        *,
        provider_type: ProviderType,
        preserve_source_files: bool = True,
    ) -> None:
        """
        Write compressed companion data for all batch input files when supported.

        :param provider_type: Provider type for the batch payload.
        :param preserve_source_files: Whether raw batch-in .jsonl files are preserved after compression.
        """
        if provider_type != "openai":
            return

        try:
            batch_in_dir = self._orch._fm.path_batch_in()
            for fpath in sorted(batch_in_dir.glob("*.jsonl")):
                compress_file_to_zip(
                    fpath,
                    preserve_source_file=preserve_source_files,
                )
        except Exception:
            pass

    def _recompress_outputs(
        self,
        *,
        preserve_source_files: bool = True,
    ) -> None:
        """
        Recompress batch output files in batch-out.

        :param preserve_source_files: Whether raw .jsonl outputs are preserved after compression.
        :return: None.
        """
        try:
            batch_out_dir = self._orch._fm.path_batch_out()

            for zip_path in sorted(batch_out_dir.glob("*.zip")):
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        names = zf.namelist()
                        if not names:
                            continue
                        inner_fname = names[0]
                        payload = zf.read(inner_fname).decode("utf-8")
                    persist_to_zip(
                        zip_path,
                        payload,
                        inner_fname=inner_fname,
                    )
                except (OSError, zipfile.BadZipFile, RuntimeError, UnicodeDecodeError):
                    continue

            for fpath in sorted(batch_out_dir.glob("*.jsonl")):
                try:
                    compress_file_to_zip(
                        fpath,
                        preserve_source_file=preserve_source_files,
                    )
                except (OSError, zipfile.BadZipFile, RuntimeError):
                    continue
        except Exception:
            pass

    def _vacuum(
        self,
        *,
        provider_type: Optional[ProviderType] = None,
        preserve_source_files: bool = True,
        vacuum_sqlite: bool = True,
        transfer_metadata: bool = True,
        transfer_batch_pending: bool = True,
    ) -> None:
        """
        Run disk space saving tasks for batch-related artifacts.

        :param provider_type: Provider type for batch input compression. Defaults to the current provider.
        :param preserve_source_files: Whether raw batch input/output files are preserved after compression.
        :param vacuum_sqlite: Whether to run SQLite VACUUM after cleanup.
        :param transfer_metadata: Whether to transfer supported metadata to parquet.
        :param transfer_batch_pending: Whether to archive inactive batch_pending rows to parquet.
        :return: None.
        """
        if provider_type is None:
            provider_type = self._orch._provider.provider_type

        self._compress_inputs(
            provider_type=provider_type,
            preserve_source_files=preserve_source_files,
        )
        self._recompress_outputs(preserve_source_files=preserve_source_files)

        datastore = self._orch._backend._get_datastore()
        if transfer_metadata and hasattr(datastore, "_transfer_metadata_to_parquet"):
            try:
                datastore._transfer_metadata_to_parquet()
            except Exception:
                pass

        if transfer_batch_pending:
            _archive_inactive_batch_pending(datastore, delete_after_transfer=True)

        if vacuum_sqlite and hasattr(datastore, "_vacuum"):
            try:
                datastore._vacuum()
            except Exception:
                pass

    def forget_batch(
        self,
        batch_uuid,
        provider_type: ProviderType,
        *,
        cancel: bool = False,
    ) -> None:
        """
        Forget a batch of calls, removing them from the cache and preventing them from being used in future calls.

        :param batch_uuid: The batch UUID to forget
        :param provider_type: The provider type (e.g., 'openai', 'google')
        :param cancel: Whether to cancel the batch with the provider before forgetting locally.
        """
        if not self._orch._provider.is_compatible(provider_type):
            raise ValueError(
                f"Given provider '{provider_type}' is not compatible with current '{self._orch._provider.provider_type}'"
            )

        if cancel:
            if not isinstance(self._orch._provider, BatchProvider):
                raise TypeError("Current provider does not support batch cancellation.")
            self._orch._provider.cancel_batch(batch_uuid, provider_type=provider_type)

        # Get the datastore from the backend
        datastore = self._orch._backend._get_datastore()
        datastore.clear_batch_pending(batch_uuid)

    def _import_batch(
        self, batch_uuid, path_to_batch, *, provider_type: ProviderType
    ) -> None:
        """
        Not implemented yet - do not use.
        """
        # Note: fundamental limitation, because the batch is the _output_; there is no way to retrieve the
        # corresponding input (doc_hash) alone. Unless there is some way to guarantee the custom_id is the
        # doc_hash? I suppose that you should be allowed to manually import a file
        # if it has otherwise been run through and stored into SQLite. But then why do you need to do it
        # manually at all? This function is on hold.
        raise NotImplementedError

        if not self._orch._provider.is_compatible(provider_type):
            raise ValueError(
                f"Given provider '{provider_type}' is not compatible with current '{self._orch._provider.provider_type}'"
            )

        backend: "BatchProvider" = self._orch._provider
        with open(path_to_batch, "r") as f:
            content = f.read()
        batch_result = backend.decode_batch_content(content)
        datastore = self._orch._backend._get_datastore()

        # this requires the input to already be stored in the "pending"
        datastore.store_ready_batch(batch_result)
