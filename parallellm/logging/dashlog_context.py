from parallellm.logging.dash_logger import DashboardLogger


class DashboardLoggerContext:
    def __init__(self, logger: "DashboardLogger"):
        self.logger = logger
        self.prev_value = None

    def __enter__(self):
        self.prev_value = self.logger.display
        self.logger.set_display(True)
        return self.logger

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.set_display(self.prev_value)
        return False
