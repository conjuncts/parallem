from parallellm.logging.dash_logger import DashboardLogger


class DashboardLoggerContext:
    def __init__(self, logger: "DashboardLogger"):
        self.logger = logger

    def __enter__(self):
        self.logger.set_display(True)
        return self.logger

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.set_display(False)
        return False
