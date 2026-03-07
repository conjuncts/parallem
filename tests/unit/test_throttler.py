import time
import threading
from pipelinellm.core.throttler import Throttler


class TestThrottler:
    def test_no_throttling_when_disabled(self):
        """Test that no delay is applied when throttling is disabled"""
        throttler = Throttler(max_requests_per_window=None)

        # Should return 0 delay for any number of rapid requests
        for _ in range(10):
            delay = throttler.calculate_delay()
            assert delay == 0.0

        assert not throttler.is_enabled()

    def test_throttling_allows_requests_under_limit(self):
        """Test that requests under the limit are allowed without delay"""
        throttler = Throttler(max_requests_per_window=5, window_seconds=1.0)

        # First 5 requests should have no delay
        for i in range(5):
            delay = throttler.calculate_delay()
            assert delay == 0.0, f"Request {i + 1} should not be throttled"

        # 6th request should be throttled
        delay = throttler.calculate_delay()
        assert delay > 0, "6th request should be throttled"

    def test_throttling_delay_calculation(self):
        """Test that throttling delay is calculated correctly"""
        throttler = Throttler(max_requests_per_window=5, window_seconds=1.0)

        # Fill up the request limit
        for _ in range(5):
            throttler.calculate_delay()

        # Next request should be delayed
        delay = throttler.calculate_delay()

        # Delay should be positive and reasonable (less than 1 second)
        assert 0 < delay <= 1.0

    def test_old_timestamps_cleanup(self):
        """Test that old timestamps are properly cleaned up"""
        throttler = Throttler(max_requests_per_window=5, window_seconds=1.0)

        # Manually add old timestamps
        old_time = time.time() - 2.0  # 2 seconds ago (outside 1-second window)
        for i in range(3):
            throttler.record_request(old_time + i * 0.1)

        # Current request count should be 0 after cleanup
        current_count = throttler.get_current_request_count()
        assert current_count == 0

    def test_mixed_old_and_recent_timestamps(self):
        """Test cleanup with mix of old and recent timestamps"""
        throttler = Throttler(max_requests_per_window=5, window_seconds=1.0)

        current_time = time.time()
        old_time = current_time - 2.0  # 2 seconds ago (outside window)
        recent_time = current_time - 0.5  # 0.5 seconds ago (inside window)

        # Add mix of old and recent timestamps
        throttler.record_request(old_time)
        throttler.record_request(recent_time)
        throttler.record_request(current_time)

        # Only recent timestamps should remain
        current_count = throttler.get_current_request_count()
        assert current_count == 2  # recent_time and current_time should remain

    def test_throttling_window_configuration(self):
        """Test that throttling window can be configured"""
        throttler = Throttler(max_requests_per_window=10, window_seconds=0.5)

        config = throttler.get_config()
        assert config["window_seconds"] == 0.5
        assert config["max_requests_per_minute"] == 10
        assert config["enabled"] is True

    def test_thread_safety_of_throttling(self):
        """Test that throttling is thread-safe"""
        throttler = Throttler(max_requests_per_window=10, window_seconds=1.0)

        def make_requests():
            for _ in range(3):
                throttler.calculate_delay()

        threads = [threading.Thread(target=make_requests) for _ in range(3)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # If we get here without deadlock, basic thread safety is working
        assert True

    def test_record_request_manual(self):
        """Test manually recording requests"""
        throttler = Throttler(max_requests_per_window=3, window_seconds=1.0)

        # Record requests manually
        current_time = time.time()
        throttler.record_request(current_time)
        throttler.record_request(current_time + 0.1)
        throttler.record_request(current_time + 0.2)

        # Should have 3 requests now
        assert throttler.get_current_request_count() == 3

        # Next calculate_delay should be throttled since we're at limit
        delay = throttler.calculate_delay()
        assert delay > 0

    def test_get_config_comprehensive(self):
        """Test get_config returns all expected fields"""
        throttler = Throttler(max_requests_per_window=5, window_seconds=1.0)

        # Make some requests
        for _ in range(3):
            throttler.calculate_delay()

        config = throttler.get_config()

        assert "max_requests_per_minute" in config
        assert "window_seconds" in config
        assert "current_request_count" in config
        assert "enabled" in config

        assert config["max_requests_per_minute"] == 5
        assert config["window_seconds"] == 1.0
        assert config["current_request_count"] == 3
        assert config["enabled"] is True

    def test_disabled_throttler_config(self):
        """Test config for disabled throttler"""
        throttler = Throttler(max_requests_per_window=None)

        config = throttler.get_config()
        assert config["max_requests_per_minute"] is None
        assert config["enabled"] is False
        assert config["current_request_count"] == 0
