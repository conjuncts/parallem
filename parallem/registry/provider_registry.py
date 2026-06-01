import threading
from dataclasses import dataclass
from typing import Callable, Optional

from parallem.provider.base import BaseProvider


ProviderFactory = Callable[[Optional[object]], BaseProvider]
"""
A callable that accepts a single positional ``client`` argument and returns a
:class:`~parallem.provider.base.BaseProvider` instance.

If the provider does not need a client, the factory should accept and ignore
the argument (``lambda client: MyProvider()``).

A bare *class* whose constructor signature is ``__init__(self, client)`` also
satisfies this protocol.
"""


@dataclass
class _ProviderEntry:
    sync: Optional[ProviderFactory] = None
    concurrent: Optional[ProviderFactory] = None
    batch: Optional[ProviderFactory] = None


_registry: dict[str, _ProviderEntry] = {}
_lock = threading.Lock()


def register_provider(
    name: str,
    *,
    sync: Optional[ProviderFactory] = None,
    concurrent: Optional[ProviderFactory] = None,
    batch: Optional[ProviderFactory] = None,
) -> None:
    """Register a custom provider under *name*.

    At least one strategy factory must be supplied.  A factory is any callable
    ``(client) -> BaseProvider``; a class with ``__init__(self, client)`` works
    directly.

    :param name: Unique identifier for the provider (e.g. ``"myprovider"``).
        Used in :class:`~parallem.types.LLMIdentity` strings like
        ``"myprovider/my-model"``.
    :param sync: Factory for the synchronous strategy.
    :param concurrent: Factory for the concurrent (async) strategy.
    :param batch: Factory for the batch strategy.
    :raises ValueError: If *name* is empty or collides with a built-in provider
        name, or if no factory is provided.
    :raises TypeError: If a supplied factory is not callable.
    """
    if not name:
        raise ValueError("Provider name must be a non-empty string.")

    _builtin_names = {"openai", "anthropic", "google", "bedrock", "multi"}
    if name in _builtin_names:
        raise ValueError(f"'{name}' is a built-in provider name and cannot be overridden.")

    if sync is None and concurrent is None and batch is None:
        raise ValueError(
            "At least one strategy factory (sync, concurrent, or batch) must be provided."
        )

    for label, factory in (
        ("sync", sync),
        ("concurrent", concurrent),
        ("batch", batch),
    ):
        if factory is not None and not callable(factory):
            raise TypeError(f"'{label}' factory must be callable, got {type(factory)!r}.")

    with _lock:
        _registry[name] = _ProviderEntry(sync=sync, concurrent=concurrent, batch=batch)


def unregister_provider(name: str) -> None:
    """Remove a previously registered provider.

    :param name: Provider name to remove.
    :raises KeyError: If *name* is not in the registry.
    """
    with _lock:
        if name not in _registry:
            raise KeyError(f"No provider registered under '{name}'.")
        del _registry[name]


def is_registered(name: str) -> bool:
    """Return whether *name* is a registered custom provider.

    :param name: Provider name to check.
    :return: ``True`` if a provider with *name* exists in the registry.
    """
    return name in _registry


def get_provider(
    name: str,
    strategy: str,
    *,
    client: Optional[object] = None,
) -> BaseProvider:
    """Instantiate a registered provider for *strategy*.

    :param name: Provider name.
    :param strategy: One of ``"sync"``, ``"concurrent"``, or ``"batch"``.
    :param client: Optional pre-initialised client to pass to the factory.
    :return: A :class:`~parallem.provider.base.BaseProvider` instance.
    :raises KeyError: If *name* is not registered.
    :raises NotImplementedError: If the requested *strategy* has no factory.
    """
    with _lock:
        entry = _registry.get(name)

    if entry is None:
        raise KeyError(f"No provider registered under '{name}'.")

    factory = getattr(entry, strategy, None)
    if factory is None:
        raise NotImplementedError(f"Provider '{name}' does not support the '{strategy}' strategy.")

    return factory(client)
