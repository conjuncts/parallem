from colorama import Fore, Style, init
import logging

# Initialize colorama
init()


class ColoredFormatter(logging.Formatter):
    _status_colors = {
        logging.INFO: Fore.CYAN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.DEBUG: Fore.LIGHTBLACK_EX,
    }

    def format(self, record):
        # Apply colors based on log level
        color = self._status_colors.get(record.levelno, "")
        record.levelname = f"{color}[{record.levelname}]{Style.RESET_ALL}"

        return super().format(record)


class DashboardAwareHandler(logging.StreamHandler):
    """
    A logging handler that coordinates with the dashboard logger when active.
    When the dashboard is active, it prepends \r\033[K to clear the current line
    before outputting log messages.
    """

    def __init__(self, dashlog, stream=None):
        super().__init__(stream)
        self._dashlog = dashlog

    def emit(self, record):
        """Emit a log record, coordinating with dashboard if active."""
        try:
            msg = self.format(record)

            # Check if dashboard logger is active and displaying
            if self._dashlog.display and self._dashlog._console_written:
                # Clear current line and print message
                self.stream.write(f"\r\033[K{msg}\n")
                self.stream.flush()

            else:
                # No dashboard or dashboard not active, use regular output
                self.stream.write(msg + self.terminator)
                self.stream.flush()

        except Exception:
            self.handleError(record)


# Configure logging to output to console with colors - only for pllm loggers
_pllm_log_handler = None


def get_pllm_log_handler(dashlog) -> DashboardAwareHandler:
    global _pllm_log_handler
    if _pllm_log_handler is None:
        _pllm_log_handler = DashboardAwareHandler(dashlog)
        _pllm_log_handler.setFormatter(ColoredFormatter("%(levelname)s %(message)s"))
    return _pllm_log_handler
