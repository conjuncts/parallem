"""
Throttler class for rate limiting requests using a rolling window approach.
"""

import time
import threading
from collections import deque
from typing import Optional


class Throttler:
    """
    A rate limiter that uses a rolling window to enforce request limits.

    This throttler maintains a rolling window of request timestamps to enforce
    rate limits like "no more than N requests per minute". When the limit is reached,
    it calculates how long to wait until the oldest request in the window expires.
    """

    def __init__(
        self,
        max_requests_per_window: Optional[int] = None,
        window_seconds: float = 60.0,
    ):
        """
        Initialize the Throttler.

        :param max_requests_per_minute: Rate limit for requests (None = no limit)
        :param window_seconds: Time window for rate limiting in seconds
        """
        self._max_requests_per_window = max_requests_per_window
        self._window_seconds = window_seconds
        self._request_timestamps = deque()  # Rolling window of submission timestamps
        self._lock = threading.Lock()

    def _cleanup_old_timestamps(self, current_time: float) -> None:
        """Remove timestamps outside the throttling window"""
        cutoff_time = current_time - self._window_seconds
        while self._request_timestamps and self._request_timestamps[0] <= cutoff_time:
            self._request_timestamps.popleft()

    def calculate_delay(self) -> float:
        """
        Calculate how long to wait before submitting the next request.

        :return: Delay in seconds (0 if no throttling is needed)
        """
        if self._max_requests_per_window is None:
            return 0.0

        current_time = time.time()

        with self._lock:
            # Clean up old timestamps
            self._cleanup_old_timestamps(current_time)

            # Check if we're at the rate limit
            if len(self._request_timestamps) < self._max_requests_per_window:
                # Record this request and proceed
                self._request_timestamps.append(current_time)
                return 0.0

            # We're at the limit - calculate delay until oldest request expires
            oldest_timestamp = self._request_timestamps[0]
            delay = (oldest_timestamp + self._window_seconds) - current_time
            return max(0.0, delay)

    def record_request(self, timestamp: Optional[float] = None) -> None:
        """
        Record a request timestamp.

        :param timestamp: Timestamp to record (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            self._cleanup_old_timestamps(timestamp)
            self._request_timestamps.append(timestamp)

    def is_enabled(self) -> bool:
        """
        Check if throttling is enabled.

        :return: True if throttling is enabled, False otherwise
        """
        return self._max_requests_per_window is not None

    def get_current_request_count(self) -> int:
        """
        Get the current number of requests in the window.

        :return: Number of requests in the current window
        """
        if not self.is_enabled():
            return 0

        current_time = time.time()
        with self._lock:
            self._cleanup_old_timestamps(current_time)
            return len(self._request_timestamps)

    def get_config(self) -> dict:
        """
        Get current throttler configuration.

        :return: Dictionary with current throttling settings
        """
        with self._lock:
            return {
                "max_requests_per_minute": self._max_requests_per_window,
                "window_seconds": self._window_seconds,
                "current_request_count": len(self._request_timestamps),
                "enabled": self.is_enabled(),
            }
