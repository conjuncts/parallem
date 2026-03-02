import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Union
from parallellm.core.backend import BaseBackend
from parallellm.core.datastore.sqlite import SQLiteDatastore
from parallellm.core.response import BatchLLMResponse
from parallellm.core.file_manager import FileManager
from parallellm.logging.dash_logger import (
    DashboardLogger,
    HashStatus,
    PrimitiveDashboardLogger,
)
from parallellm.provider.base import BatchProvider
from parallellm.types import (
    BatchIdentifier,
    BatchResult,
    CallIdentifier,
    CommonQueryParameters,
    CohortIdentifier,
    LLMIdentity,
    ParsedResponse,
)


@dataclass
class _BatchRecord:
    """Private dataclass for organizing batch data"""

    call_ids: List[CallIdentifier] = field(default_factory=list)
    model_name: Optional[str] = None
    data: List[dict] = field(default_factory=list)
    custom_ids: List[str] = field(default_factory=list)


class BatchBackend(BaseBackend):
    """
    The batch backend is a bit different: it defers sending out requests
    (if the value is needed, NotAvailable() is emitted),
    then upon exit, a batch is created.
    """

    def __init__(
        self,
        fm: FileManager,
        dashlog: DashboardLogger = PrimitiveDashboardLogger(),
        *,
        datastore_cls=None,
        session_id: int,
        confirm_batch_submission: bool = False,
        rewrite_cache: bool = False,
    ):
        self._fm = fm
        if datastore_cls is None:
            self._ds = SQLiteDatastore(fm)
        else:
            self._ds = datastore_cls(fm)
        self.dashlog = dashlog
        self._confirm_batch_submission = confirm_batch_submission
        self._rewrite_cache = rewrite_cache

        self._batch_buffer: list[tuple[CallIdentifier, dict]] = []
        self.session_id = session_id

        self._private_increment = 0

    def _get_datastore(self):
        return self._ds

    def submit_query(
        self,
        provider: "BatchProvider",
        params: CommonQueryParameters,
        *,
        call_id: CallIdentifier,
        **kwargs,
    ) -> BatchLLMResponse:
        """
        New control flow: Backend calls provider to get batch data, then bookkeeps it.
        This inverts control from provider calling backend.
        """

        # Get the batch call data from the provider
        stuff = provider.prepare_batch_call(
            params,
            custom_id=self.generate_custom_id(call_id),
            **kwargs,
        )

        # Bookkeep the call
        self.bookkeep_call(
            call_id,
            params["llm"],
            stuff=stuff,
        )

        return BatchLLMResponse(call_id)

    def _poll_changes(self, call_id: CallIdentifier):
        """
        For batch, not our responsibility
        """

    def retrieve(
        self, call_id: CallIdentifier, metadata=False
    ) -> Optional[ParsedResponse]:
        return self._ds.retrieve(call_id, metadata=metadata)

    def close(self):
        """Clean up resources"""
        if hasattr(self._ds, "close"):
            self._ds.close()

    def __del__(self):
        """Clean up resources when the SyncBackend is destroyed"""
        try:
            self.close()
        except Exception:
            # Ignore errors during cleanup in destructor
            pass

    def bookkeep_call(
        self,
        call_id: CallIdentifier,
        llm: LLMIdentity,
        stuff: dict,
    ):
        """
        Saves a call so that it may be executed as a batch later

        :param call_id: The call identifier
        :param stuff: Arbitrary data needed to make the call later. Should be a dict.
        """
        # If call is already in a pending batch,
        # do not add it again
        if self._ds.is_call_in_pending_batch(call_id):
            return

        self._batch_buffer.append((call_id, llm.model_name, stuff))

    def generate_custom_id(
        self,
        call_id: CallIdentifier,
    ):
        """Generate a custom ID for batch calls

        f"{agent_name}-{session_id}-{seq_id}-{counter}", which is guaranteed to be unique unless a major
        error has occurred
        """
        agent_name = call_id["agent_name"]
        seq_id = call_id["seq_id"]
        custom_id = f"{agent_name}-{self.session_id}-{seq_id}-{self._private_increment}"
        self._private_increment += 1
        return custom_id

    def execute_batch(
        self,
        provider: "BatchProvider",
        dl: DashboardLogger,
        *,
        max_batch_size=1000,
        partition_by_model_name=True,
    ) -> CohortIdentifier:
        """Execute the batch of calls"""

        if not self._batch_buffer:
            return CohortIdentifier(batch_ids=[], session_id=self.session_id)

        # 1. Split based on model_name
        # Many batch APIs (like OpenAI's) require the same model across the entire batch
        index_groups: list[list[int]] = []  # list of groups of indices
        if partition_by_model_name:
            mn2b = {}
            for i, (call_id, model_name, stuff) in enumerate(self._batch_buffer):
                if model_name not in mn2b:
                    mn2b[model_name] = []
                mn2b[model_name].append(i)
            index_groups = list(mn2b.values())
        else:
            index_groups = [list(range(len(self._batch_buffer)))]

        # 2. Split into groups of max_batch_size
        _bdr = []
        for batch in index_groups:
            if len(batch) <= max_batch_size:
                _bdr.append(batch)
            else:
                for i in range(0, len(batch), max_batch_size):
                    _bdr.append(batch[i : i + max_batch_size])
        index_groups = _bdr

        # dict with keys: call_ids, model_name, stuff, custom_ids
        batches: list[_BatchRecord] = []
        for index_gp in index_groups:
            rowwise = _BatchRecord()
            for i in index_gp:
                call_id, model_name, stuff = self._batch_buffer[i]
                rowwise.call_ids.append(call_id)
                rowwise.model_name = model_name
                rowwise.data.append(stuff)
            rowwise.custom_ids = provider.get_batch_custom_ids(rowwise.data)
            batches.append(rowwise)

        pending_fpaths = []

        # Ask for confirmation if requested
        _saved_already = False
        if self._confirm_batch_submission:
            total_calls = sum(len(batch) for batch in index_groups)
            num_batches = len(index_groups)

            confirmed = dl.confirm_batch_submission(num_batches, total_calls)

            if confirmed == "p":
                # Preview: write them to file, but do not yet submit
                _saved_already = True
                for record in batches:
                    fpath = self._fm.save_batch_in(record.data)
                    pending_fpaths.append(fpath)
                dl.print(f"Batch preview files written to {pending_fpaths[0]}")
                confirmed = dl.confirm_batch_submission(
                    num_batches, total_calls, allow_preview=False
                )

            if confirmed == "n":
                dl.print("Batch submission cancelled by user.")
                # Don't clear the buffer - allow the user to try again later
                return CohortIdentifier(batch_ids=[], session_id=self.session_id)
            # else, proceed
        if not _saved_already:
            for record in batches:
                fpath = self._fm.save_batch_in(record.data)
                pending_fpaths.append(fpath)

        # 3. Submit each batch
        batch_ids = []
        for fpath, record in zip(pending_fpaths, batches):
            batch_uuid = provider.submit_batch_to_provider(fpath, record.model_name)
            ident = BatchIdentifier(
                call_ids=record.call_ids,
                custom_ids=record.custom_ids,
                batch_uuid=batch_uuid,
            )
            self._ds.store_pending_batch(ident)
            batch_ids.append(ident)

            # Log batch submission to dashboard
            dl.update_hash(batch_uuid, HashStatus.SENT_BATCH)
            dl.print("Sent batch:", ident.batch_uuid)
            dl._update_console()

        cohort_id = CohortIdentifier(batch_ids=batch_ids, session_id=self.session_id)
        # Clear the batch buffer after execution
        self._batch_buffer.clear()
        return cohort_id

    def persist(self):
        """Persist any remaining data and datastore"""
        self._ds.persist()

    def persist_to_zip(
        self, stuff: Union[str, list[dict]], fpath: Path, *, inner_fname: str = None
    ) -> None:
        """
        Helper function to persist anything to a zip file.

        :param stuff: If str, writes the string as a single file.
        :param fpath: The path to the zip file to create.
        :param inner_fname: The name of the file inside the zip.
            If not given, then `fpath` minus ".zip".
        """
        import zipfile

        with zipfile.ZipFile(fpath, "w") as zf:
            if inner_fname is None:
                inner_fname = fpath.stem
            if isinstance(stuff, str):
                zf.writestr(inner_fname, stuff)
            else:
                # write a jsonl
                with zf.open(inner_fname + ".jsonl", "w") as f:
                    for item in stuff:
                        line = json.dumps(item) + "\n"
                        f.write(line.encode("utf-8"))

    # TODO: Also need to store the batch_uuid
    # In that datastore, we need to store
    # (doc_hash, session_id, seq_id, batch_uuid)
    # so probably the easiest way is to add a possibly NULL column (batch_uuid)
    # to the existing SQL database.

    # But that requires response (TEXT) to be not null.
    # Hence, we will need to create a new table "unready_batch_responses"
    # That contains (agent_name, seq_id, session_id, doc_hash, provider_type, batch_uuid)

    def download_batch_from_provider(
        self,
        provider: "BatchProvider",
        batch_uuid: str,
        *,
        save_to_disk: Literal[None, "zip"] = "zip",
    ) -> List[BatchResult]:
        """
        Given a batch UUID, download the results.

        This is a helper function that calls the provider's download_batch_results method.

        The list can contain both ready and error results.
        Empty list = still pending.

        :param batch_uuid: The UUID of the batch to download.
        :param save_to_disk: If "zip", saves the results to a zip file.
            If None, does not save to disk.
        """

        batch_results = provider.download_batch(batch_uuid)

        for res in batch_results:
            if save_to_disk == "zip":
                ending = ".zip" if res.status == "ready" else "_err.zip"
                batch_fname = os.path.basename(batch_uuid)
                fpath = self._fm.path_batch_out() / f"{batch_fname}{ending}"
                self.persist_to_zip(
                    res.raw_output, fpath=fpath, inner_fname=batch_uuid + ".jsonl"
                )

            if res.status == "ready":
                self._ds.store_ready_batch(res, upsert=self._rewrite_cache)
                # Log batch storage to dashboard
                # self._ds.clear_batch_pending(batch_uuid)
            else:
                # TODO
                # self._ds.store_error_batch(res)
                # self._ds.clear_batch_pending(batch_uuid)
                pass
        return batch_results

    def try_download_all_batches(
        self,
        provider: "BatchProvider",
        dl: DashboardLogger,
    ):
        """
        Try to download all batches and clean up completed ones
        """
        pending_batches = self._ds.get_all_pending_batch_uuids()

        statuses = {
            "pending": 0,
            "ready": 0,
            "error": 0,
        }
        for batch_uuid, batch_provider in pending_batches:
            if not provider.is_compatible(batch_provider):
                continue
            batch_results = self.download_batch_from_provider(
                provider, batch_uuid, save_to_disk="zip"
            )
            for batch_result in batch_results:
                if batch_result.status == "ready":
                    dl.update_hash(batch_uuid, HashStatus.STORED_BATCH)
                    dl.print(f"Batch {batch_uuid} completed and stored.")
                    # Clean up the pending batch record
                    self._ds.clear_batch_pending(batch_uuid)
                    statuses["ready"] += 1
                elif batch_result.status == "error":
                    dl.update_hash(batch_uuid, HashStatus.STORED_ERROR_BATCH)
                    dl.print(f"Batch {batch_uuid} completed with errors and stored.")
                    # Clean up the pending batch record even for errors
                    self._ds.clear_batch_pending(batch_uuid)
                    statuses["error"] += 1

            if not batch_results:
                dl.update_hash(batch_uuid, HashStatus.SENT_BATCH)
                dl.print(f"Batch {batch_uuid} is still pending.")
                statuses["pending"] += 1
        return statuses


class DebugBatchBackend(BatchBackend):
    """
    Mimics the BatchBackend behavior.
    Except having to wait for the batch to complete, upon persist(),
    values are actually generated synchronously, so that the batch always "finishes" immediately.
    Useful for testing.
    """
