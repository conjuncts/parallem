import pytest

from parallem.provider.base import SyncProvider, ConcurrentProvider, BatchProvider
from parallem.registry.provider_registry import (
    get_provider,
    is_registered,
    register_provider,
    unregister_provider,
)
from parallem.provider.multi.provider_selector import dynamic_select_provider


# ---------------------------------------------------------------------------
# Minimal stub providers used across tests
# ---------------------------------------------------------------------------


class _StubSync(SyncProvider):
    provider_type = "stub"

    def __init__(self, client=None):
        self.client = client

    def get_default_llm_identity(self):
        raise NotImplementedError

    def validate_request_compatibility(self, params, **kwargs):
        pass

    def parse_response(self, raw_response, provider_type=None):
        raise NotImplementedError

    def prepare_sync_call(self, params, **kwargs):
        raise NotImplementedError


class _StubConcurrent(ConcurrentProvider):
    provider_type = "stub"

    def __init__(self, client=None):
        self.client = client

    def get_default_llm_identity(self):
        raise NotImplementedError

    def validate_request_compatibility(self, params, **kwargs):
        pass

    def parse_response(self, raw_response, provider_type=None):
        raise NotImplementedError

    def prepare_concurrent_call(self, params, **kwargs):
        raise NotImplementedError


class _StubBatch(BatchProvider):
    provider_type = "stub"

    def __init__(self, client=None):
        self.client = client

    def get_default_llm_identity(self):
        raise NotImplementedError

    def validate_request_compatibility(self, params, **kwargs):
        pass

    def parse_response(self, raw_response, provider_type=None):
        raise NotImplementedError

    def prepare_batch_call(self, params, custom_id, **kwargs):
        raise NotImplementedError

    def get_batch_custom_ids(self, stuff, provider_type):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cleanup():
    """Remove any test providers registered during a test."""
    yield
    for name in ("stubprovider", "synconly", "factorybased"):
        if is_registered(name):
            unregister_provider(name)


# ---------------------------------------------------------------------------
# register_provider – validation
# ---------------------------------------------------------------------------


def test_register_empty_name_raises():
    with pytest.raises(ValueError, match="non-empty"):
        register_provider("", sync=_StubSync)


def test_register_builtin_name_raises():
    with pytest.raises(ValueError, match="built-in"):
        register_provider("openai", sync=_StubSync)


def test_register_no_factories_raises():
    with pytest.raises(ValueError, match="At least one"):
        register_provider("stubprovider")


def test_register_non_callable_factory_raises():
    with pytest.raises(TypeError, match="callable"):
        register_provider("stubprovider", sync="not_a_callable")


# ---------------------------------------------------------------------------
# register_provider / is_registered / unregister_provider
# ---------------------------------------------------------------------------


def test_register_and_is_registered():
    register_provider("stubprovider", sync=_StubSync)
    assert is_registered("stubprovider")


def test_is_registered_false_for_unknown():
    assert not is_registered("does_not_exist")


def test_unregister_removes_provider():
    register_provider("stubprovider", sync=_StubSync)
    unregister_provider("stubprovider")
    assert not is_registered("stubprovider")


def test_unregister_unknown_raises():
    with pytest.raises(KeyError, match="stubprovider"):
        unregister_provider("stubprovider")


def test_register_overwrites_existing():
    register_provider("stubprovider", sync=_StubSync)

    class _AltSync(_StubSync):
        pass

    register_provider("stubprovider", sync=_AltSync)
    provider = get_provider("stubprovider", "sync")
    assert isinstance(provider, _AltSync)


# ---------------------------------------------------------------------------
# get_provider
# ---------------------------------------------------------------------------


def test_get_provider_sync_class():
    register_provider("stubprovider", sync=_StubSync)
    provider = get_provider("stubprovider", "sync")
    assert isinstance(provider, _StubSync)


def test_get_provider_passes_client():
    sentinel = object()
    register_provider("stubprovider", sync=_StubSync)
    provider = get_provider("stubprovider", "sync", client=sentinel)
    assert provider.client is sentinel


def test_get_provider_factory_callable():
    sentinel = object()
    instances = []

    def _factory(client):
        inst = _StubSync(client=client)
        instances.append(inst)
        return inst

    register_provider("factorybased", sync=_factory)
    provider = get_provider("factorybased", "sync", client=sentinel)
    assert len(instances) == 1
    assert provider.client is sentinel


def test_get_provider_unknown_raises():
    with pytest.raises(KeyError, match="does_not_exist"):
        get_provider("does_not_exist", "sync")


def test_get_provider_unsupported_strategy_raises():
    register_provider("synconly", sync=_StubSync)
    with pytest.raises(NotImplementedError, match="concurrent"):
        get_provider("synconly", "concurrent")


def test_get_provider_all_strategies():
    register_provider(
        "stubprovider",
        sync=_StubSync,
        concurrent=_StubConcurrent,
        batch=_StubBatch,
    )
    assert isinstance(get_provider("stubprovider", "sync"), _StubSync)
    assert isinstance(get_provider("stubprovider", "concurrent"), _StubConcurrent)
    assert isinstance(get_provider("stubprovider", "batch"), _StubBatch)


# ---------------------------------------------------------------------------
# Integration with dynamic_select_provider
# ---------------------------------------------------------------------------


def test_dynamic_select_provider_uses_registry():
    register_provider("stubprovider", sync=_StubSync)
    provider = dynamic_select_provider("stubprovider", "sync")
    assert isinstance(provider, _StubSync)


def test_dynamic_select_provider_unregistered_raises():
    with pytest.raises(NotImplementedError):
        dynamic_select_provider("totally_unknown", "sync")


# ---------------------------------------------------------------------------
# Top-level package exports
# ---------------------------------------------------------------------------


def test_top_level_exports():
    import parallem

    assert callable(parallem.register_provider)
    assert callable(parallem.unregister_provider)
