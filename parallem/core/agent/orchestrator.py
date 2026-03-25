from logging import Logger
from typing import List, Literal, Optional
from parallem.core.agent.agent import AgentContext
from parallem.core.backend import BaseBackend
from parallem.core.batch_namespace import BatchNamespace
from parallem.core.state.non_msg_state import NonMessageState
from parallem.logging.dashlog_context import DashboardLoggerContext
from parallem.provider.base import BaseProvider
from parallem.core.file_manager import FileManager
from parallem.logging.dash_logger import DashboardLogger
from parallem.types import AskParameters, LLMResponse

import polars as pl


class AgentOrchestrator:
    """The AgentOrchestrator manages and creates agents. It's how you begin using pipelinellm."""

    def __init__(
        self,
        file_manager: FileManager,
        backend: BaseBackend,
        provider: BaseProvider,
        *,
        logger: Logger,
        dashlog: DashboardLogger,
        ask_params: Optional[AskParameters] = None,
        ignore_cache: bool = False,
        strategy: Optional[Literal["sync", "async", "batch"]] = None,
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
        """
        self._backend = backend
        self._fm = file_manager
        self._provider = provider
        self._logger = logger
        self._batch = BatchNamespace(self)
        self._userdata = NonMessageState(self._backend)

        # dashlog's display is disabled by default
        self._dashlog: DashboardLogger = dashlog

        self.ask_params = ask_params or {}
        self.ignore_cache = ignore_cache
        self.strategy = strategy

    def __enter__(self):
        """Enter the context manager, returning self."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager, automatically calling persist()."""
        self.persist()
        return False

    def agent(
        self,
        name: str = "",
        *,
        ask_params: Optional[AskParameters] = None,
    ) -> AgentContext:
        """
        Constructs an agent.

        pipelinellm does things a bit differently.
        While typically an agent is associated with a single LLM,
        pipelinellm identifies an agent with a process, program, or algorithm
        which itself can ask LLMs questions, but also functions, MCP servers, and humans.
        It just so happens that the agent uses LLM(s) to automate much of its decision making.
        """
        if ask_params is None:
            ask_params = self.ask_params

        return AgentContext(
            str(name),
            self,
            ask_params=ask_params,
            ignore_cache=self.ignore_cache,
        )

    @property
    def userdata(self):
        return self._userdata

    def persist(self):
        """
        Ensure that everything is properly saved AND cleans up resources.
        """

        if getattr(self._backend, "execute_batch", None):
            with self.dashboard():
                # Must print this one
                self._backend.execute_batch(self._provider, dl=self._dashlog)
                self._dashlog._update_console()
                self._dashlog.finalize_line()
        elif self._dashlog.display:
            self._dashlog._update_console()
            self._dashlog.finalize_line()

        self._backend.persist()
        self._fm.persist()

    def save_to_file(
        self,
        responses: List[LLMResponse],
        fname: str,
        *,
        format: str = "batch-openai",
    ):
        """
        Save a list of responses to a file.
        Creates an openai batch file.
        """
        raise NotImplementedError

    def get_session_counter(self):
        """
        Get session counter, aka session ID."""
        return self._fm._get_session_counter()

    def dashboard(self, *, keep_when_done=True):
        """
        Context manager for activating a dashlog only for a specific block of code.
        """
        return DashboardLoggerContext(self._dashlog, keep_when_done=keep_when_done)

    @property
    def batch(self) -> BatchNamespace:
        """Namespace for batch operations."""
        return self._batch

    def export_tables(
        self,
        directory: Optional[str],
        *,
        filetype: Literal["csv", "tsv", "parquet"] = "parquet",
    ):
        """
        Export all tables from the backend to files.

        :param directory: Directory to export tables to. If None, uses the default datastore directory.
        :param filetype: Export file type - "polars" or "parquet" for parquet files, "csv" for CSV, "tsv" for TSV.
        """
        return self._backend._get_datastore().export_tables(
            directory, filetype=filetype
        )

    def export_polars(
        self,
    ) -> dict[str, "pl.DataFrame"]:
        """
        Export all tables from the backend as Polars DataFrames.

        :returns: A dictionary mapping table names to Polars DataFrames.
        """
        return self._backend._get_datastore().export_polars()

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
