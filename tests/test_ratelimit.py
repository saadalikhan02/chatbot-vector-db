"""Unit tests for src/chatbot_rag/ratelimit.py - pure logic, no I/O."""

from __future__ import annotations

import time

from chatbot_rag.ratelimit import RateLimiter


class TestRateLimiter:
    def test_allows_requests_under_the_limit(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        assert limiter.allow("key") is True
        assert limiter.allow("key") is True
        assert limiter.allow("key") is True

    def test_blocks_requests_over_the_limit(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.allow("key") is True
        assert limiter.allow("key") is True
        assert limiter.allow("key") is False

    def test_keys_are_independent(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.allow("key-a") is True
        assert limiter.allow("key-b") is True
        assert limiter.allow("key-a") is False
        assert limiter.allow("key-b") is False

    def test_window_expiry_allows_requests_again(self):
        limiter = RateLimiter(max_requests=1, window_seconds=0.05)
        assert limiter.allow("key") is True
        assert limiter.allow("key") is False
        time.sleep(0.06)
        assert limiter.allow("key") is True

    def test_blocked_request_is_not_recorded(self):
        # A blocked call must not itself consume a slot - otherwise a
        # burst of rejected requests could keep the window permanently
        # full even after it should have expired.
        limiter = RateLimiter(max_requests=1, window_seconds=0.05)
        assert limiter.allow("key") is True
        for _ in range(5):
            assert limiter.allow("key") is False
        time.sleep(0.06)
        assert limiter.allow("key") is True
