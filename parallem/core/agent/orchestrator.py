import asyncio
from logging import Logger
import polars as pl
from typing import Literal, Optional
from parallem.core.agent.agent import AgentContext
from parallem.core.backend import BaseBackend
from parallem.core.backend.batch_backend import BatchBackend
from parallem.core.batch_namespace import BatchNamespace
from parallem.core.datastore.input_storage import InputStorage
from parallem.core.exception import NotAvailable, ParallemSignal, PendingNotAvailable
from parallem.core.export_namespace import ExportNamespace
from parallem.core.seq_id_store import InMemorySeqIdStore, SeqIdStore
from parallem.provider.base import BaseProvider
from parallem.core.file_manager import FileManager
from parallem.logging.dash_logger import DashboardLogger
from parallem.types import AskParameters


class AgentOrchestrator:
    """The AgentOrchestrator manages and creates agents. It's how you begin using parallem."""

    def __init__(
        self,
        file_manager: FileManager,
        backend: BaseBackend,
        provider: BaseProvider,
        *,
        input_storage: InputStorage = None,
        logger: Logger,
        dashlog: DashboardLogger,
        ask_params: Optional[AskParameters] = None,
        ignore_cache: bool = False,
        strategy: Optional[Literal["sync", "async", "batch"]] = None,
        seq_id_store: Optional[SeqIdStore] = None,
        error_mode: Optional[str] = None,
    ):
        """
        Initialize the AgentOrchestrator.

        :param file_manager: File manager for handling persistence and metadata
        :param backend: Backend for data storage and retrieval
        :param provider: Provider for submitting queries to LLM APIs
        :param logger: Logger instance
        :param dashlog: Dashboard logger for pretty printing hash status
        :param ask_params: Default parameters for ask_llm() calls
        :param ignore_cache: If True, always submit to the API instead of using cached responses
        :param seq_id_store: Optional sequence ID store implementation
        """
        self._backend = backend
        self._fm = file_manager
        self._provider = provider
        if input_storage is None:
            input_storage = InputStorage(self._fm)
        self._input_storage = input_storage
        self._logger = logger
        self._batch = BatchNamespace(self)
        self._export_ns = ExportNamespace(self)

        # dashlog's display is disabled by default
        self._dashlog: DashboardLogger = dashlog

        self.ask_params = ask_params or {}
        self.ignore_cache = ignore_cache
        self.strategy = strategy
        self._seq_id_store = seq_id_store or InMemorySeqIdStore()
        self._error_mode = error_mode

    def __enter__(self):
        """Enter the context manager, returning self."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager, automatically calling persist()."""
        if exc_value is None or isinstance(exc_value, ParallemSignal):
            self.finalize_tasks()
        self._backend.persist()
        if self._input_storage is not None:
            self._input_storage.persist()
        self._fm.persist()
        if isinstance(exc_value, ParallemSignal):
            # We suppress it to allow graceful exits.
            if exc_type is NotAvailable:
                self._logger.debug("Exited due to unavailable values (no action needed).")
            elif exc_type is PendingNotAvailable:
                self._logger.debug("Exited due to values still in a pending batch (no action needed).")
            return True
        return False

    def agent(
        self,
        name: str = "",
        *,
        ask_params: Optional[AskParameters] = None,
    ) -> AgentContext:
        """
        Constructs an agent.

        ParaLLeM identifies an agent with a process, program, or algorithm
        which itself can ask LLMs questions, but also functions, MCP servers, and humans.
        It just so happens that the agent uses LLM(s) to automate much of its decision making.
        """
        if ask_params is None:
            ask_params = self.ask_params
        else:
            ask_params = {**self.ask_params, **ask_params}

        return AgentContext(
            str(name),
            self,
            ask_params=ask_params,
            ignore_cache=self.ignore_cache,
            error_mode=self._error_mode,
        )

    def finalize_tasks(self):
        """
        Run any remaining tasks.
            - In async mode, waits for all pending agents to complete.
            - In batch mode, executes the entire batch.
            - In sync mode, does nothing since agents are executed immediately.
        """

        if isinstance(self._backend, BatchBackend) and self.strategy == "batch":
            with self.dashboard():
                # Must print this one
                self._backend.submit_all_batches(self._provider, dl=self._dashlog)
                self._dashlog._update_console()
                self._dashlog.finalize_line()
        elif self._dashlog.display:
            self._dashlog._update_console()
            self._dashlog.finalize_line()

    def submit_and_close(self):
        """
        Ensure that everything is properly finalized, saved, and resources cleaned up.

        Crucially, submits all pending batches.
        """
        self.finalize_tasks()
        self._backend.persist()
        self._fm.persist()
        if self._input_storage is not None:
            self._input_storage.persist()

    def get_session_counter(self):
        """
        Get session counter, aka session ID."""
        return self._fm._get_session_counter()

    def dashboard(self, *, keep_when_done=True):
        """
        Context manager for activating a dashlog only for a specific block of code.
        """
        return self._dashlog.context(keep_when_done=keep_when_done)

    @property
    def batch(self) -> BatchNamespace:
        """Namespace for batch operations."""
        return self._batch

    @property
    def export(self) -> ExportNamespace:
        """Namespace for export operations."""
        return self._export_ns

    def import_polars(
        self,
        tables: dict[str, "pl.DataFrame"],
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
        return self._backend._get_datastore().import_polars(tables, update=update)

    def next_seq_id(self, agent_name: str) -> int:
        """
        Allocate the next sequence ID for an agent within this session.

        :param agent_name: Agent name.
        :return: Next sequence ID for this agent in the current session.
        """
        session_id = self.get_session_counter()
        return self._seq_id_store.next_seq_id(session_id, agent_name)

    async def gather(self, *coros_or_futures, return_exceptions: bool = False):
        """
        Gather a list of coroutines.

        Main distinction between this and asyncio.gather() is that
            this will catch NotAvailable and PendingNotAvailable exceptions.

        If asyncio.gather() must be used, it should be wrapped 

        :param coros: List of coroutines to gather.
        :return: Results of the gathered coroutines.
        """
        # Wrap coroutines such that NotAvailable and PendingNotAvailable exceptions are caught.
        async def safe_coro(coro):
            try:
                return await coro
            except (NotAvailable, PendingNotAvailable):
                return None

        results = await asyncio.gather(
            *(safe_coro(c) for c in coros_or_futures),
            return_exceptions=return_exceptions
        )
        return results
