import logging
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from parallem.core.agent.orchestrator import AgentOrchestrator
from parallem.core.file_manager import FileManager
from parallem.logging.dash_logger import DashboardLogger
from parallem.logging.fancy import get_pllm_log_handler
from parallem.provider.multi.provider_selector import dynamic_select_provider
from parallem.types import AskParameters, HashByOptions, LLMIdentity, MinorTweaks

if TYPE_CHECKING:
    from parallem.core.throttler import Throttler


def resume_directory(
    directory,
    *,
    strategy: Literal["sync", "concurrent", "batch", "hybrid"] = "sync",
    provider: Literal["openai", "google", "anthropic", "multi"] = "multi",
    datastore: Literal["sqlite", "sqlite_parquet"] = "sqlite",
    dry_run=False,
    log_level=logging.INFO,
    ignore_cache=False,
    rewrite_cache=False,
    throttler: Optional["Throttler"] = None,
    tweaks: MinorTweaks = MinorTweaks(),
    dashboard: bool = False,
    client: Optional[Any] = None,
    hash_by: HashByOptions = None,
    save_input: Optional[bool] = None,
    llm: Union[LLMIdentity, str, None] = None,
) -> AgentOrchestrator:
    """
    Resume an AgentOrchestrator from a previously saved directory.

    :param directory: Path to directory

    :param strategy: Execution strategy for LLM calls
    :param provider: LLM provider to use for API calls. "multi" allows a mixture of providers.
    :param datastore: Backend datastore type for response storage. Recommended: sqlite.
    :param dry_run: If True, validate setup without making actual API calls
    :param log_level: Logging level for the session
    :param ignore_cache: If True, always submit to API instead of using cached responses
    :param rewrite_cache: If True, overwrite cached responses with new ones (uses upsert)
    :param throttler: Throttler instance for rate limiting (default: None, no throttling)
    :param tweaks: MinorTweaks instance for fine-tuning behavior
    :param dashboard: If True, pretty prints sent requests in real time
    :param client: Optional pre-initialized client instance (ie. OpenAI, Google, Anthropic, etc.)

    :param hash_by: By default, responses with identical content but different configs
        are considered equivalent. Specify additional parameters (like "llm")
        to differentiate.
    :param save_input: By default, input documents are not saved. Set to True to save them.
    :param llm: By default, LLM identity to use for API calls.

    :return: Configured AgentOrchestrator instance
    :raises ValueError: If strategy is not supported
    :raises NotImplementedError: If dry_run is True or strategy is not implemented
    """

    # Logic to resume from the specified directory
    # 1. Validation
    if strategy not in ["sync", "concurrent", "batch", "hybrid"]:
        raise ValueError(f"Unknown strategy '{strategy}'")
    if dry_run:
        raise NotImplementedError("Dry run is not implemented yet")
    if isinstance(tweaks, dict):
        tweaks = MinorTweaks(**tweaks)
    # 2. Setup logger
    dashlog = DashboardLogger(k=10, display=dashboard)
    pllm_log_handler = get_pllm_log_handler(dashlog)

    logger = logging.getLogger("parallem")
    logger.setLevel(log_level)
    logger.addHandler(pllm_log_handler)
    logger.debug("Resuming directory")

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    # 3. Setup components
    fm = FileManager(directory)

    logger.debug("Creating backend")
    if datastore == "sqlite":
        datastore_cls = None  # default

    if strategy == "concurrent":
        from parallem.core.backend.concurrent_backend import ConcurrentBackend

        backend = ConcurrentBackend(
            fm,
            dashlog=dashlog,
            datastore_cls=datastore_cls,
            rewrite_cache=rewrite_cache,
            max_concurrent=tweaks.max_concurrent,
            throttler=throttler,
        )
    elif strategy == "sync":
        from parallem.core.backend.sync_backend import SyncBackend

        backend = SyncBackend(
            fm,
            dashlog=dashlog,
            datastore_cls=datastore_cls,
            rewrite_cache=rewrite_cache,
            throttler=throttler,
        )
    elif strategy == "batch":
        from parallem.core.backend.batch_backend import BatchBackend

        backend = BatchBackend(
            fm,
            dashlog=dashlog,
            datastore_cls=datastore_cls,
            session_id=fm._get_session_counter(),
            confirm_batch_submission=tweaks.batch_user_confirmation,
            max_batch_size=tweaks.batch_max_size,
            rewrite_cache=rewrite_cache,
        )
    else:
        raise NotImplementedError(f"Strategy '{strategy}' is not implemented yet")

    logger.debug("Creating provider")
    provider_obj = dynamic_select_provider(
        provider, strategy, multi_allowed=True, client=client
    )

    logger.debug("Creating AgentOrchestrator")

    ask_params: AskParameters = {}
    if hash_by is not None:
        ask_params["hash_by"] = hash_by
    if save_input is not None:
        ask_params["save_input"] = save_input
    if llm is not None:
        ask_params["llm"] = llm
    bm = AgentOrchestrator(
        file_manager=fm,
        backend=backend,
        provider=provider_obj,
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
            statuses = backend.try_download_all_batches(provider_obj, d)
            # Don't store these statuses: batch_hash != msg_hash
            d.clear(clear_console=False)
        d.finalize_line()

        if statuses["pending"] > 0 and tweaks.batch_wait_until_complete:
            # TODO: handle this better
            print("Cannot proceed until all batches are complete.")
            exit(0)

    return bm
