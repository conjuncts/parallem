from logging import Logger
import asyncio
from concurrent.futures import Future
import inspect
import polars as pl
from typing import Any, Callable, Coroutine, List, Literal, Optional, Union
from parallem.core.agent.agent import AgentContext
from parallem.core.backend import BaseBackend
from parallem.core.batch_namespace import BatchNamespace
from parallem.core.exception import NotAvailable, ParallemSignal, PendingNotAvailable
from parallem.core.state.non_msg_state import NonMessageState
from parallem.logging.dashlog_context import DashboardLoggerContext
from parallem.provider.base import BaseProvider
from parallem.core.file_manager import FileManager
from parallem.logging.dash_logger import DashboardLogger
from parallem.types import AskParameters, LLMResponse
from parallem.utils.manip import reduce_to_list


class AgentOrchestrator:
    """The AgentOrchestrator manages and creates agents. It's how you begin using parallem."""

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
        strategy: Optional[Literal["sync", "concurrent", "batch"]] = None,
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
        self._pending_agent_coroutines: list[
            tuple[Coroutine[Any, Any, Any], Future[Any]]
        ] = []

    def create_agent(
        self,
        fn: Callable[..., Any],
        *fn_args,
        agent_name: str = "",
        ask_params: Optional[AskParameters] = None,
        **fn_kwargs,
    ) -> Future[Any]:
        """
        Create a future-like handle for an agent function.

        - sync/batch: executes coroutine agents immediately.
        - concurrent/async: queues coroutine agents to run together.
        """
        promise: Future[Any] = Future()

        agt = self.agent(agent_name, ask_params=ask_params)
        agent_coro: Optional[Coroutine[Any, Any, Any]] = None

        is_coro = inspect.iscoroutinefunction(fn)
        if not is_coro:
            # Then it's a regular function
            if self.strategy in ["sync", "batch"]:
                # If sync mode = execute immediately and set result on promise
                try:
                    result = fn(agt, *fn_args, **fn_kwargs)
                except Exception as exc:
                    promise.set_exception(exc)
                    return promise
                promise.set_result(result)
                return promise
            else:
                # If concurrent mode, turn regular function into coroutine and queue
                async def _coro_wrapper():
                    return fn(agt, *fn_args, **fn_kwargs)

                agent_coro = _coro_wrapper()
        else:
            # Then it's an async function
            if self.strategy in ["sync", "batch"]:
                # If sync mode = execute immediately and set result on promise
                try:
                    agent_coro = fn(agt, *fn_args, **fn_kwargs)
                    resolved = asyncio.run(agent_coro)
                except Exception as exc:
                    promise.set_exception(exc)
                    return promise
                promise.set_result(resolved)
                return promise
            agent_coro = fn(agt, *fn_args, **fn_kwargs)

        if self.strategy != "concurrent" or agent_coro is None:
            promise.set_exception(
                ValueError(
                    f"Invalid strategy for create_agent: {self.strategy}. "
                    "Expected one of 'sync', 'batch', or 'concurrent'."
                )
            )
            return promise

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._pending_agent_coroutines.append((agent_coro, promise))
            return promise

        task = loop.create_task(agent_coro)

        def _done_callback(done_task):
            if promise.done():
                return
            if done_task.cancelled():
                promise.cancel()
                return
            exc = done_task.exception()
            if exc is not None:
                promise.set_exception(exc)
                return
            promise.set_result(done_task.result())

        task.add_done_callback(_done_callback)
        return promise

    def run_agents(
        self,
        handle: Union[Future, List[Future]],
        *handles: Future[Any],
        return_exceptions: bool = None,
    ):
        """
        Resolve multiple created agent handles.

        Similar to gather semantics:
        - Executes all queued concurrent coroutines first.
        - Collects all outcomes.
        - If ``return_exceptions`` is False, raises once at the end if any handle failed,
          prioritizing ParallemSignal subclasses (e.g., NotAvailable).
        - If ``return_exceptions`` is True, returns a list of outcomes and exceptions.
          (Matches behavior of asyncio.gather with return_exceptions=True)
        """
        if self.strategy == "concurrent" and self._pending_agent_coroutines:
            self._run_pending_agents()

        outcomes: list[Any] = []
        first_signal: Optional[BaseException] = None
        first_other: Optional[BaseException] = None

        handles = reduce_to_list(handle, list(handles))
        for handle in handles:
            try:
                outcome = handle.result()
            except BaseException as exc:
                outcomes.append(exc)
                if isinstance(exc, ParallemSignal):
                    if first_signal is None:
                        first_signal = exc
                elif first_other is None:
                    first_other = exc
                continue
            outcomes.append(outcome)

        if return_exceptions is True:
            return outcomes

        if first_signal is not None:
            first_signal._from_run_agents = True
            raise first_signal
        if first_other is not None:
            raise first_other
        return outcomes

    def _run_pending_agents(self):
        if not self._pending_agent_coroutines:
            return

        pending = list(self._pending_agent_coroutines)
        self._pending_agent_coroutines.clear()

        async def _runner():
            coros = [coro for coro, _ in pending]
            return await asyncio.gather(*coros, return_exceptions=True)

        results = asyncio.run(_runner())
        for (_, handle), outcome in zip(pending, results):
            if handle.done():
                continue
            if isinstance(outcome, Exception):
                handle.set_exception(outcome)
            else:
                handle.set_result(outcome)

    def __enter__(self):
        """Enter the context manager, returning self."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager, automatically calling persist()."""
        self.persist()
        if isinstance(exc_value, ParallemSignal):
            # If the signal was emitted by run_agents, we suppress it to allow graceful exits.
            if exc_value._from_run_agents:
                if exc_type is NotAvailable:
                    self._logger.info("Exited due to unavailable values.")
                elif exc_type is PendingNotAvailable:
                    self._logger.info("Exited due to values still in a pending batch.")
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

        parallem does things a bit differently.
        While typically an agent is associated with a single LLM,
        parallem identifies an agent with a process, program, or algorithm
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

        if self.strategy == "concurrent":
            self._run_pending_agents()

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
