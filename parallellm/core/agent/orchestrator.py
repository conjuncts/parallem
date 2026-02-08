from logging import Logger
from typing import List, Literal, Optional, Union
from parallellm.core.agent.agent import AgentContext
from parallellm.core.backend import BaseBackend
from parallellm.core.msg.state import MessageState
from parallellm.core.hydrate import hydrate_llm_response, hydrate_msg_state
from parallellm.provider.base import BaseProvider
from parallellm.file_io.file_manager import FileManager
from parallellm.logging.dash_logger import DashboardLogger
from parallellm.types import AskParameters, LLMResponse


class AgentOrchestrator:
    def __init__(
        self,
        file_manager: FileManager,
        backend: BaseBackend,
        provider: BaseProvider,
        *,
        logger: Logger,
        dashlog: DashboardLogger,
        special_dashlog: DashboardLogger = None,
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
        :param special_dashlog: Dashboard logger for the beginning/end of lifecycle.
        :param ask_params: Default parameters for ask_llm() calls
        :param ignore_cache: If True, always submit to the API instead of using cached responses
        """
        self._backend = backend
        self._fm = file_manager
        self._provider = provider
        self._logger = logger

        # dashlog's display is disabled by default
        self._dashlog: DashboardLogger = dashlog
        if special_dashlog is None:
            special_dashlog = dashlog
        self.special_dashlog: DashboardLogger = special_dashlog

        self.ask_params = ask_params or {}
        self.ignore_cache = ignore_cache
        self.strategy = strategy

    def __enter__(self):
        """Enter the context manager, returning self."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager, automatically calling persist()."""
        if self._dashlog.display:
            self._dashlog._update_console()

            self._dashlog.finalize_line()
        self.persist()
        return False

    def agent(
        self,
        name: str = "",
        *,
        ask_params: Optional[AskParameters] = None,
    ):
        """
        Delineates an agent.
        """
        if ask_params is None:
            ask_params = self.ask_params

        return AgentContext(
            name,
            self,
            ask_params=ask_params,
            ignore_cache=self.ignore_cache,
        )

    def get_msg_state(self, agent: AgentContext) -> MessageState:
        """
        Load the MessageState for a specific agent.
        """
        msg_state = self._fm.load_agent_msg_state(agent.agent_name)
        msg_state._true_agent = agent
        msg_state = hydrate_msg_state(msg_state, self._backend)
        return msg_state

    def save_msg_state(self, agent: AgentContext, msg_state: MessageState):
        """
        Save the MessageState for a specific agent.
        """
        self._fm.save_agent_msg_state(agent.agent_name, msg_state)

    def save_userdata(self, key, value):
        """
        The intended way to let data persist across runs
        """
        return self._fm.save_userdata(key, value)

    def load_userdata(self, key):
        """
        The intended way to let data persist across runs
        """
        data = self._fm.load_userdata(key)

        # If the loaded data is an LLMResponse, inject the backend and hydrate
        data = hydrate_llm_response(data, self._backend)

        return data

    def persist(self):
        """
        Ensure that everything is properly saved AND cleans up resources.
        """
        self._backend.persist()

        if getattr(self._backend, "execute_batch", None):
            special_dl = self.special_dashlog
            special_dl.set_display(True)
            self._backend.execute_batch(self._provider, special_dl=special_dl)
            special_dl._update_console()
            special_dl.set_display(False, clear_console=False)

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
