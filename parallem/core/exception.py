class ParallemSignal(Exception):
    """
    parallem uses exceptions as "signals" to prevent
    code from executing.

    They should always be automatically caught as long as you are using
    the BatchManager inside a 'with' block.
    """

    _from_run_agents = False
    "Flag for whether it was emitted by AgentOrchestrator.run_agents"


class NotAvailable(ParallemSignal):
    pass


class PendingNotAvailable(NotAvailable):
    """
    Raised when a response is already pending in a batch and should not be resent.
    """

    pass


class IntegrityError(Exception):
    """
    If you are seeing this exception, something changed between runs
    """

    pass


class ProviderCompatibilityError(Exception):
    """
    Raised when a provider determines that a given request is not compatible with its capabilities.
    """

    pass
