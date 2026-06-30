class ParallemSignal(BaseException):
    """
    parallem uses exceptions as "signals" to prevent
    code from executing.

    They should always be automatically caught as long as you are using
    the BatchManager inside a 'with' block.
    """


class NotAvailable(ParallemSignal):

    def __init__(
        self,
        message="NotAvailable: this response will be available once its batch is processed."
    ):
        super().__init__(message)



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
