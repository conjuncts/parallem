import logging
from typing import Literal, Optional

from parallellm.core.agent.orchestrator import AgentOrchestrator
from parallellm.file_io.file_manager import FileManager
from parallellm.logging.dash_logger import DashboardLogger
from parallellm.logging.fancy import get_parallellm_log_handler
from parallellm.provider.openai.sdk import BatchOpenAIProvider
from parallellm.types import HashByOptions, MinorTweaks, ProviderType


class ParalleLLM:
    @staticmethod
    def resume_directory(
        directory,
        *,
        strategy: Literal["sync", "async", "batch", "hybrid"] = "sync",
        provider: ProviderType = "openai",
        datastore: Literal["sqlite", "sqlite_parquet"] = "sqlite",
        dry_run=False,
        log_level=logging.INFO,
        ignore_cache=False,
        rewrite_cache=False,
        throttler=None,
        tweaks: MinorTweaks = MinorTweaks(),
        dashboard: bool = False,
        hash_by: HashByOptions = None,
        save_input: bool = None,
    ) -> AgentOrchestrator:
        """
        Resume an AgentOrchestrator from a previously saved directory.

        :param directory: Path to directory

        :param strategy: Execution strategy for LLM calls
        :param provider: LLM provider to use for API calls
        :param datastore: Backend datastore type for response storage. Recommended: sqlite.
        :param dry_run: If True, validate setup without making actual API calls
        :param log_level: Logging level for the session
        :param ignore_cache: If True, always submit to API instead of using cached responses
        :param rewrite_cache: If True, overwrite cached responses with new ones (uses upsert)
        :param throttler: Throttler instance for rate limiting (default: None, no throttling)
        :param tweaks: MinorTweaks instance for fine-tuning behavior
        :param dashboard: If True, pretty prints sent requests in real time

        :param hash_by: By default, responses with identical content but different configs
            are considered equivalent. Specify additional parameters (like "llm")
            to differentiate.
        :param save_input: By default, input documents are not saved. Set to True to save them.

        :return: Configured AgentOrchestrator instance
        :raises ValueError: If strategy is not supported
        :raises NotImplementedError: If dry_run is True or strategy is not implemented
        """

        # Logic to resume from the specified directory
        # 1. Validation
        if strategy not in ["sync", "async", "batch", "hybrid"]:
            raise ValueError(f"Unknown strategy '{strategy}'")
        if dry_run:
            raise NotImplementedError("Dry run is not implemented yet")
        if isinstance(tweaks, dict):
            tweaks = MinorTweaks(**tweaks)
        # 2. Setup logger
        dashlog = DashboardLogger(k=10, display=dashboard)
        parallellm_log_handler = get_parallellm_log_handler(dashlog)

        logger = logging.getLogger("parallellm")
        logger.setLevel(log_level)
        logger.addHandler(parallellm_log_handler)
        logger.debug("Resuming directory")

        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False

        # 3. Setup components
        fm = FileManager(directory)

        logger.debug("Creating backend")
        if datastore == "sqlite":
            datastore_cls = None  # default

        if strategy == "async":
            from parallellm.core.backend.async_backend import AsyncBackend

            backend = AsyncBackend(
                fm,
                dashlog=dashlog,
                datastore_cls=datastore_cls,
                rewrite_cache=rewrite_cache,
                max_concurrent=tweaks.async_max_concurrent,
                throttler=throttler,
            )
        elif strategy == "sync":
            from parallellm.core.backend.sync_backend import SyncBackend

            backend = SyncBackend(
                fm,
                dashlog=dashlog,
                datastore_cls=datastore_cls,
                rewrite_cache=rewrite_cache,
                throttler=throttler,
            )
        elif strategy == "batch":
            from parallellm.core.backend.batch_backend import BatchBackend

            backend = BatchBackend(
                fm,
                dashlog=dashlog,
                datastore_cls=datastore_cls,
                session_id=fm._get_session_counter(),
                confirm_batch_submission=tweaks.batch_user_confirmation,
                rewrite_cache=rewrite_cache,
            )
        else:
            raise NotImplementedError(f"Strategy '{strategy}' is not implemented yet")

        logger.debug("Creating provider")
        if provider == "openai":
            from parallellm.provider.openai.sdk import (
                AsyncOpenAIProvider,
                SyncOpenAIProvider,
            )

            if strategy == "async":
                from openai import AsyncOpenAI

                client = AsyncOpenAI()
                provider = AsyncOpenAIProvider(client=client)
            elif strategy == "batch":
                from openai import OpenAI

                client = OpenAI()
                provider = BatchOpenAIProvider(client=client)
            else:
                # For other strategies, default to sync for now
                from openai import OpenAI

                client = OpenAI()
                provider = SyncOpenAIProvider(client=client)
        elif provider == "google":
            from parallellm.provider.google.sdk import (
                AsyncGoogleProvider,
                BatchGoogleProvider,
                SyncGoogleProvider,
            )
            from google import genai

            if strategy == "async":
                client = genai.Client()
                provider = AsyncGoogleProvider(client=client)
            elif strategy == "batch":
                client = genai.Client()
                provider = BatchGoogleProvider(client=client)
            else:
                client = genai.Client()
                provider = SyncGoogleProvider(client=client)
        elif provider == "anthropic":
            from parallellm.provider.anthropic.sdk import (
                AsyncAnthropicProvider,
                SyncAnthropicProvider,
            )

            if strategy == "async":
                from anthropic import AsyncAnthropic

                client = AsyncAnthropic()
                provider = AsyncAnthropicProvider(client=client)
            else:
                from anthropic import Anthropic

                client = Anthropic()
                provider = SyncAnthropicProvider(client=client)
        else:
            raise NotImplementedError(f"Provider '{provider}' not implemented yet")

        logger.debug("Creating AgentOrchestrator")

        ask_params = {}
        if hash_by is not None:
            ask_params["hash_by"] = hash_by
        if save_input is not None:
            ask_params["save_input"] = save_input
        bm = AgentOrchestrator(
            file_manager=fm,
            backend=backend,
            provider=provider,
            logger=logger,
            dashlog=dashlog,
            ignore_cache=ignore_cache,
            strategy=strategy,
            ask_params=ask_params,
        )

        logger.info(f"Resuming with session_id={bm.get_session_counter()}")

        # try downloading previous batches if any
        if strategy == "batch":
            with bm.dashboard() as d:
                statuses = backend.try_download_all_batches(provider, d)
                # Don't store these statuses: batch_hash != msg_hash
                d.clear(clear_console=False)
            d.finalize_line()

            if statuses["pending"] > 0 and tweaks.batch_wait_until_complete:
                # TODO: handle this better
                print("Cannot proceed until all batches are complete.")
                exit(0)

        return bm


resume_directory = ParalleLLM.resume_directory
