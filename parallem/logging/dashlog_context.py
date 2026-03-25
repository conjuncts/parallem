from parallem.logging.dash_logger import DashboardLogger


class DashboardLoggerContext:
    def __init__(self, logger: "DashboardLogger", *, keep_when_done=True):
        """
        Context manager to display dashboard only for a specific block of code.

        :param logger: DashboardLogger instance.
        :param keep_when_done: Whether to keep whatever ways displayed after leaving the context
        """
        self.logger = logger
        self.prev_value = None
        self.keep_when_done = keep_when_done

    def __enter__(self):
        self.prev_value = self.logger.display
        self.logger.set_display(True)
        return self.logger

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.set_display(self.prev_value)
        if not self.keep_when_done:
            self.logger.clear(clear_console=True)
        return False
