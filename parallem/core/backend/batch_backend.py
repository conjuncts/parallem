import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union
from parallem.core.backend import BaseBackend
from parallem.core.datastore.input_storage import InputStorage
from parallem.core.compress.pack_zip import compress_file_to_zip, persist_to_zip
from parallem.core.datastore.sqlite import SQLiteDatastore
from parallem.core.exception import PendingNotAvailable
from parallem.core.response import BatchLLMResponse
from parallem.core.file_manager import FileManager
from parallem.logging.dash_logger import (
    DashboardLogger,
    HashStatus,
    PrimitiveDashboardLogger,
)
from parallem.provider.base import BatchProvider
from parallem.types import (
    BatchIdentifier,
    BatchResult,
    CallIdentifier,
    CommonQueryParameters,
    CohortIdentifier,
    LLMIdentity,
    ParsedResponse,
)


@dataclass
class BatchGroup:
    """Private dataclass for organizing batch data"""

    call_ids: List[CallIdentifier]
    llm: Optional[LLMIdentity]
    data: List[dict]
    custom_ids: List[str]


@dataclass
class BatchBufferItem:
    call_id: CallIdentifier
    llm: LLMIdentity
    stuff: dict


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
        max_batch_size: int = 1000,
        compress_inputs: bool = False,
        rewrite_cache: bool = False,
    ):
        self._fm = fm
        if datastore_cls is None:
            self._ds = SQLiteDatastore(fm)
        else:
            self._ds = datastore_cls(fm)
        self._input_storage = InputStorage(fm)
        self.dashlog = dashlog
        self._confirm_batch_submission = confirm_batch_submission
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self._max_batch_size = max_batch_size
        self._compress_inputs = compress_inputs
        self._rewrite_cache = rewrite_cache

        self._batch_buffer: list[BatchBufferItem] = []
        """List of BatchBufferItem objects"""
        self.session_id = session_id

        self._private_increment = 0
        self._pending_count = 0
        """Count of requests that were already in pending batches"""

    def _get_datastore(self):
        return self._ds

    def _get_input_storage(self):
        return self._input_storage

    def ask_llm_and_store(
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

        # Check if the call is already in a pending batch
        if self._ds.is_call_in_pending_batch(call_id):
            self._pending_count += 1
            raise PendingNotAvailable()

        provider.validate_request_compatibility(
            params,
            **kwargs,
        )

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

    def retrieve(self, call_id: CallIdentifier, metadata=False) -> Optional[ParsedResponse]:
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

        self._batch_buffer.append(BatchBufferItem(call_id=call_id, llm=llm, stuff=stuff))

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
        max_batch_size: Optional[int] = None,
        partition_by_model_name=True,
    ) -> CohortIdentifier:
        """Execute the batch of calls"""

        if max_batch_size is None:
            max_batch_size = self._max_batch_size

        if not self._batch_buffer:
            return CohortIdentifier(batch_ids=[], session_id=self.session_id)

        # 1. Split based on model_name
        # Many batch APIs (like OpenAI's) require the same model across the entire batch
        # one "flight" = group of calls, less than 1000 calls, sent together in one batch API call
        grouped_calls: list[list[BatchBufferItem]] = []
        if partition_by_model_name:
            llm_to_indices: dict[LLMIdentity, list[BatchBufferItem]] = {}
            # group_by based on LLMIdentity
            for item in self._batch_buffer:
                if item.llm not in llm_to_indices:
                    llm_to_indices[item.llm] = []
                llm_to_indices[item.llm].append(item)
            grouped_calls = list(llm_to_indices.values())
        else:
            grouped_calls = [self._batch_buffer]

        # 2. Split into groups of max_batch_size
        _result = []
        for gp in grouped_calls:
            if len(gp) <= max_batch_size:
                _result.append(gp)
            else:
                for i in range(0, len(gp), max_batch_size):
                    _result.append(gp[i : i + max_batch_size])
        grouped_calls = _result

        # dict with keys: call_ids, model_name, stuff, custom_ids
        batch_groups: list[BatchGroup] = []
        for gp in grouped_calls:
            if not gp:
                continue  # should not happen, but just in case
            llm_common = None
            call_ids = []
            data = []

            for item in gp:
                llm_common = item.llm  # all calls should have same llm
                call_ids.append(item.call_id)
                data.append(item.stuff)

            assert llm_common is not None
            custom_ids = provider.get_batch_custom_ids(data, provider_type=llm_common.provider_type)
            batch_groups.append(
                BatchGroup(
                    call_ids=call_ids,
                    llm=llm_common,
                    data=data,
                    custom_ids=custom_ids,
                )
            )

        pending_fpaths = []

        # Ask for confirmation if requested
        _saved_already = False
        if self._confirm_batch_submission:
            total_calls = sum(len(batch.call_ids) for batch in batch_groups)
            num_batches = len(batch_groups)

            confirmed = dl.confirm_batch_submission(num_batches, total_calls)

            if confirmed == "p":
                # Preview: write them to file, but do not yet submit
                _saved_already = True
                for record in batch_groups:
                    fpath = self._fm.save_batch_in(record.data)
                    pending_fpaths.append(fpath)
                dl.info(f"Batch preview files written to {pending_fpaths[0]}")
                confirmed = dl.confirm_batch_submission(
                    num_batches, total_calls, allow_preview=False
                )

            if confirmed == "n":
                dl.info("Batch submission cancelled by user.")
                # Don't clear the buffer - allow the user to try again later
                return CohortIdentifier(batch_ids=[], session_id=self.session_id)
            # else, proceed
        if not _saved_already:
            for record in batch_groups:
                fpath = self._fm.save_batch_in(record.data)
                pending_fpaths.append(fpath)

        # 3. Submit each batch
        batch_ids = []
        for fpath, record in zip(pending_fpaths, batch_groups):
            batch_uuid = provider.submit_batch_to_provider(fpath, record.llm)
            ident = BatchIdentifier(
                call_ids=record.call_ids,
                custom_ids=record.custom_ids,
                batch_uuid=batch_uuid,
            )
            self._ds.store_pending_batch(ident)
            batch_ids.append(ident)

            # Log batch submission to dashboard
            dl.update_hash(batch_uuid, HashStatus.SENT_BATCH)
            dl.info(f"Sent batch: {ident.batch_uuid}")
            dl._update_console()

            # Compress batch input files
            if self._compress_inputs:
                try:
                    compress_file_to_zip(
                        fpath,
                        preserve_source_file=False,
                    )
                except (
                    OSError,
                    RuntimeError,
                    zipfile.BadZipFile,
                    zipfile.LargeZipFile,
                ):
                    # Compression is an optional side effect; batch submission should still succeed.
                    pass

        cohort_id = CohortIdentifier(batch_ids=batch_ids, session_id=self.session_id)
        # Clear the batch buffer after execution
        self._batch_buffer.clear()
        return cohort_id

    def persist(self):
        """Persist any remaining data and datastore"""
        # Print summary of pending requests if any were encountered
        if self._pending_count > 0:
            self.dashlog._logger.info(
                f"Skipped {self._pending_count} request(s) already in pending batches."
            )
        self._input_storage.persist()
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
        persist_to_zip(fpath, stuff, inner_fname=inner_fname)

    def download_batch_from_provider(
        self,
        provider: "BatchProvider",
        batch_uuid: str,
        *,
        provider_type: str,
        save_to_disk: Literal[None, "jsonl", "zip"] = "zip",
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

        batch_results = provider.download_batch(batch_uuid, provider_type=provider_type)

        for res in batch_results:
            if save_to_disk == "zip" and res.raw_output is not None:
                ending = ".zip" if res.status == "ready" else "_err.zip"
                batch_fname = os.path.basename(batch_uuid)
                fpath = self._fm.path_batch_out() / f"{batch_fname}{ending}"
                self.persist_to_zip(res.raw_output, fpath=fpath, inner_fname=batch_uuid + ".jsonl")
                res.location = fpath
            elif save_to_disk == "jsonl" and res.raw_output is not None:
                ending = ".jsonl" if res.status == "ready" else "_err.jsonl"
                batch_fname = os.path.basename(batch_uuid)
                fpath = self._fm.path_batch_out() / f"{batch_fname}{ending}"
                fpath.write_text(res.raw_output, encoding="utf-8")
                res.location = fpath

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
        *,
        save_to_disk: Literal[None, "jsonl", "zip"] = "zip",
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
                provider,
                batch_uuid,
                save_to_disk=save_to_disk,
                provider_type=batch_provider,
            )
            for batch_result in batch_results:
                if batch_result.status == "ready":
                    dl.update_hash(batch_uuid, HashStatus.STORED_BATCH)
                    print(f"Batch {batch_uuid} completed and stored.")
                    # Clean up the pending batch record
                    self._ds.clear_batch_pending(batch_uuid)
                    statuses["ready"] += 1
                elif batch_result.status == "error":
                    dl.update_hash(batch_uuid, HashStatus.STORED_ERROR_BATCH)
                    print(f"Batch {batch_uuid} completed with errors and stored.")
                    if batch_result.location:
                        print(f"- Saved to {batch_result.location}")
                    # Clean up the pending batch record even for errors
                    self._ds.clear_batch_pending(batch_uuid)
                    statuses["error"] += 1

            if not batch_results:
                dl.update_hash(batch_uuid, HashStatus.SENT_BATCH)
                print(f"Batch {batch_uuid} is still pending.")
                statuses["pending"] += 1
        return statuses


class DebugBatchBackend(BatchBackend):
    """
    Mimics the BatchBackend behavior.
    Except having to wait for the batch to complete, upon persist(),
    values are actually generated synchronously, so that the batch always "finishes" immediately.
    Useful for testing.
    """
