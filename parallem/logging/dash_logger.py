import contextlib
import sys
from typing_extensions import deprecated
import shutil
import threading
from collections import OrderedDict
from typing import Literal
from colorama import Fore, Style, init
from dataclasses import dataclass
from enum import Enum

# Initialize colorama for colored output
init()


class DashboardStdout:
    """File-like wrapper used with redirect_stdout to coordinate with the dashboard.

    It writes directly to the real stdout (`sys.__stdout__`) and clears the
    dashboard line before printing user output so that interleaving looks clean.
    """

    def __init__(self, dashlog: "DashboardLogger", real_stdout):
        self._dashlog = dashlog
        self._real = real_stdout

    def write(self, s: str):
        if not s:
            return
        # If dashboard is currently shown, clear it on the real stdout first
        if self._dashlog._console_written and self._dashlog.display:
            try:
                self._real.write("\r\033[K")
                self._real.flush()
            except (OSError, ValueError):
                pass
            # After clearing, mark that console line is no longer occupied
            self._dashlog._console_written = False

        try:
            self._real.write(s)
            self._real.flush()
        except (OSError, ValueError):
            # best-effort: ignore write errors
            pass

    def flush(self):
        try:
            self._real.flush()
        except (OSError, ValueError):
            pass


class HashStatus(Enum):
    """Enum for different hash statuses"""

    CACHED = "C"  # cached
    SENT = "↗"  # sent to provider
    SENT_BATCH = "⇈"  # sent to provider in batch
    RECEIVED = "↘"  # received from provider
    RECEIVED_BATCH = "⇊"  # received from provider in batch
    STORED = "✓"  # stored in datastore
    STORED_BATCH = "✔"  # stored in datastore in batch
    STORED_ERROR_BATCH = "✖"


@dataclass
class HashEntry:
    """Data class to represent a hash entry"""

    hash_id: str
    status: HashStatus
    full_hash: str


class DashboardLogger:
    """
    Sophisticated logger that displays top k=10 hashes or batch UUIDs with dynamic console updates.

    Features:
    - Shows the top k=10 hash entries (or fewer if console width is limited)
    - Allows updating hash/batch statuses
    - Option to enable/disable console output
    - Console rewrites itself to minimize spam
    - Status changes update existing entries instead of adding new ones
    - Carriage return based rewriting that allows standard print() calls
    - Automatically adjusts display based on console width to prevent overflow
    """

    def __init__(self, k: int = 10, display: bool = True):
        """
        Initialize the DashboardLogger.

        Args:
            k: Maximum number of hashes/batches to display (default 10)
            display: Whether to display console output (default True)
        """
        self.k = k
        self.display = display
        self._lock = threading.Lock()

        # OrderedDict to maintain insertion order and allow efficient updates
        self._hashes: OrderedDict[str, HashEntry] = OrderedDict()

        # Track if we've written to console before
        self._console_written = False

        self._context_depth = 0
        self._context_prev_display: bool | None = None
        self._context_keep_when_done = True
        self._stdout_cm = None
        self._stdout_holder = None

        # Colors for different statuses
        self._status_colors = {
            HashStatus.CACHED: Fore.GREEN,
            HashStatus.SENT: Fore.CYAN,
            HashStatus.SENT_BATCH: Fore.CYAN,
            HashStatus.RECEIVED: Fore.GREEN,
            HashStatus.RECEIVED_BATCH: Fore.MAGENTA,
            HashStatus.STORED: Fore.GREEN,
            HashStatus.STORED_BATCH: Fore.GREEN,
            HashStatus.STORED_ERROR_BATCH: Fore.MAGENTA,
        }

    def __enter__(self):
        self._push_context(keep_when_done=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._pop_context(exc_type, exc_value, traceback)

    @contextlib.contextmanager
    def context(self, *, keep_when_done: bool = True):
        self._push_context(keep_when_done=keep_when_done)
        try:
            yield self
        except BaseException as exc:
            self._pop_context(type(exc), exc, exc.__traceback__)
            raise
        else:
            self._pop_context(None, None, None)

    def _get_stdout(self):
        if self._stdout_holder is not None:
            return self._stdout_holder._real
        else:
            return sys.stdout

    def _push_context(self, *, keep_when_done: bool):
        if self._context_depth == 0:
            self._context_prev_display = self.display
            self._context_keep_when_done = keep_when_done
            self.set_display(True, clear_console=False)
            self._stdout_holder = DashboardStdout(self, sys.__stdout__)
            self._stdout_cm = contextlib.redirect_stdout(self._stdout_holder)
            self._stdout_cm.__enter__()
        else:
            self._context_keep_when_done = (
                self._context_keep_when_done and keep_when_done
            )
        self._context_depth += 1

    def _pop_context(self, exc_type, exc_value, traceback):
        if self._context_depth == 0:
            return False

        self._context_depth -= 1
        if self._context_depth > 0:
            return False

        if self._stdout_cm is not None:
            self._stdout_cm.__exit__(exc_type, exc_value, traceback)
            self._stdout_cm = None
            self._stdout_holder = None

        prev_display = self._context_prev_display
        keep_when_done = self._context_keep_when_done

        self._context_prev_display = None
        self._context_keep_when_done = True

        if prev_display is not None:
            self.set_display(prev_display, clear_console=True)
        if not keep_when_done:
            self.clear(clear_console=True)
        return False

    def update_hash(self, full_hash: str, status: HashStatus):
        """
        Update or add a hash or batch UUID with the given status.

        Args:
            full_hash: The full hash string or batch UUID
            status: The status of the hash/batch
        """
        if not self.display:
            return
        with self._lock:
            # Strip "batch_" prefix if present
            full_hash = full_hash.removeprefix("msgbatch_")  # Anthropic's
            full_hash = full_hash.removeprefix("batch_")  # OpenAI's

            hash_id = full_hash[:8]  # Use first 8 characters as display ID

            if hash_id in self._hashes:
                # Update existing entry
                self._hashes[hash_id].status = status
            else:
                # Add new entry
                entry = HashEntry(hash_id=hash_id, status=status, full_hash=full_hash)
                self._hashes[hash_id] = entry

                # Keep only the most recent k entries
                while len(self._hashes) > self.k:
                    # Remove the oldest entry (first item in OrderedDict)
                    self._hashes.popitem(last=False)

            # Keep hashes stable when possible

            if self.display:
                self._update_console()

    def _update_console(self):
        """Update the console display with current hash statuses"""
        if not self.display:
            return

        # Get console width, with fallback to 80 if unable to determine
        try:
            console_width = shutil.get_terminal_size().columns
        except (OSError, ValueError):
            console_width = 80  # Fallback for environments without proper terminal

        # Build the display line with grey [DASH] prefix
        prefix = f"{Fore.LIGHTBLACK_EX}[DASH]{Style.RESET_ALL} "
        prefix_len = len("[DASH] ")  # Length without color codes

        # Calculate how many hashes we can display based on console width
        # Each hash entry is approximately: "S 12345678 " (11 characters)
        available_width = console_width - prefix_len - 5  # 5 chars buffer for safety
        max_displayable_hashes = max(1, available_width // 11)  # At least show 1 hash

        # Limit the number of hashes to display
        hashes_to_show = list(self._hashes.values())[-max_displayable_hashes:]
        if not hashes_to_show:
            # nothing to do
            return

        status_parts = []
        for entry in hashes_to_show:
            color = self._status_colors.get(entry.status, Fore.WHITE)
            status_parts.append(
                f"{color}{entry.status.value} {entry.hash_id}{Style.RESET_ALL}"
            )

        display_line = prefix + " ".join(status_parts)

        # # Additional safety check - truncate if still too long
        # if len(display_line.encode('utf-8')) > console_width:
        #     # Count visible characters (excluding color codes) and truncate
        #     visible_chars = prefix_len + sum(11 for _ in status_parts)  # 11 chars per hash entry
        #     if visible_chars > console_width:
        #         # Remove entries from the beginning until it fits
        #         while status_parts and len(prefix + " ".join(status_parts)) + prefix_len > console_width - 5:
        #             status_parts.pop(0)
        #         display_line = prefix + " ".join(status_parts)

        if self._console_written:
            # Move cursor to beginning of line and clear the entire line, then print new content
            # Use ANSI escape sequence to clear the entire line
            self._get_stdout().write(f"\r\033[K{display_line}\r")
        else:
            # First time writing - just print normally
            self._get_stdout().write(display_line)
            self._console_written = True

        self._get_stdout().flush()

    def set_display(self, display: bool, clear_console: bool = True):
        """Enable or disable console display"""
        with self._lock:
            self.display = display
            # do not show it if we are turning it on, until the next update_hash call
            if self._console_written:
                if clear_console:
                    self._get_stdout().write("\r\033[K")
                    self._get_stdout().flush()
                self._console_written = False

    def clear(self, clear_console=True):
        """Clear all hash entries"""
        with self._lock:
            self._hashes.clear()
            if self._console_written:
                if clear_console:
                    self._get_stdout().write("\r\033[K")
                    self._get_stdout().flush()
                self._console_written = False

    @deprecated("Use builtin print() directly.")
    def print(self, *args, **kwargs):
        print(*args, **kwargs)

    def finalize_line(self):
        """
        Finalize the current console line by moving to the next line.
        Call this when you want to ensure subsequent print() calls appear on new lines.
        """
        if self._console_written:
            self._get_stdout().write("\n")
            self._get_stdout().flush()
            self._console_written = False

    def ask_for_confirmation(
        self,
        prompt: str,
        valid_responses=None,
    ):
        """
        Ask the user for a yes/no confirmation, coordinating with dashboard display.

        :param prompt: The prompt message to display
        :param valid_responses: Optional set of valid responses (e.g., {'y', 'n'})
        :return: What the user responded
        """
        self.print(end="")
        response = input(prompt).strip().lower()
        while valid_responses is not None and response not in valid_responses:
            print(
                f"Invalid response. Please enter one of: {', '.join(valid_responses)}"
            )
            response = input().strip().lower()
        return response

    def confirm_batch_submission(
        self, num_batches: int, total_calls: int, allow_preview=True
    ) -> Literal["y", "n", "p"]:
        """
        Ask the user to confirm batch submission with formatted message.

        :param num_batches: Number of batches to be submitted
        :param total_calls: Total number of API calls across all batches
        :return: True if user confirms, False otherwise
        """

        # Format the message with colors
        plural = ""
        if num_batches > 1:
            plural = "es"
        message = (
            f"Submit {Fore.CYAN}{num_batches} batch{plural}{Style.RESET_ALL} "
            f"({Fore.CYAN}{total_calls} calls{Style.RESET_ALL})? (y/n): "
        )
        if allow_preview:
            message = message[:-3] + "/preview): "

        valid_responses = {"y", "n", "yes", "no"}
        if allow_preview:
            valid_responses.update({"preview", "p"})
        response = self.ask_for_confirmation(message, valid_responses=valid_responses)

        if response in {"y", "yes"}:
            return "y"
        elif response in {"preview", "p"}:
            return "p"
        else:
            return "n"


class PrimitiveDashboardLogger(DashboardLogger):
    """
    Gutted DashboardLogger that only prints
    """

    def __init__(self):
        """
        Initialize the DashboardLogger.

        Args:
            k: Maximum number of hashes/batches to display (default 10)
            display: Whether to display console output (default True)
        """
        super().__init__()

    def update_hash(self, full_hash: str, status: HashStatus):
        pass

    def _update_console(self):
        pass

    def set_display(self, display: bool):
        pass

    def clear(self):
        pass

    def finalize_line(self):
        pass

    # ask_for_confirmation and confirm_batch_submission remain unchanged
