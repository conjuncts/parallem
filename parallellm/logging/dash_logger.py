import sys
import shutil
import threading
from collections import OrderedDict
from typing import Literal
from colorama import Fore, Style, init
from dataclasses import dataclass
from enum import Enum

# Initialize colorama for colored output
init()


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
            if full_hash.startswith("batch_"):
                full_hash = full_hash[6:]  # Remove "batch_" prefix

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
            sys.stdout.write(f"\r\033[K{display_line}\r")
        else:
            # First time writing - just print normally
            sys.stdout.write(display_line)
            self._console_written = True

        sys.stdout.flush()

    def set_display(self, display: bool, clear_console: bool = True):
        """Enable or disable console display"""
        with self._lock:
            self.display = display
            # do not show it if we are turning it on, until the next update_hash call
            if self._console_written:
                if clear_console:
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()
                self._console_written = False

    def clear(self, clear_console=True):
        """Clear all hash entries"""
        with self._lock:
            self._hashes.clear()
            if self._console_written:
                if clear_console:
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()
                self._console_written = False

    def print(self, *args, **kwargs):
        """
        Print to console.
        """
        if not self.display:
            # Dashboard not active, use regular print
            print(*args, **kwargs)
            return

        with self._lock:
            # Clear the current dashboard line if it exists
            if self._console_written:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()

            # Print the user's content
            print(*args, **kwargs)

    def finalize_line(self):
        """
        Finalize the current console line by moving to the next line.
        Call this when you want to ensure subsequent print() calls appear on new lines.
        """
        if self._console_written:
            sys.stdout.write("\n")
            sys.stdout.flush()
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
        pass

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
