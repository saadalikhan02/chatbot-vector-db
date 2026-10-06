"""A small, dependency-free rate limiter for the serving layer (api.py).

Sliding-window, in-memory, per-key. Deliberately not a third-party
dependency (e.g. slowapi) or a distributed store (e.g. Redis) - this
project has consistently avoided adding external services for things a
small, well-scoped implementation covers (see retrieval.py's reasoning
for not using a vector database at this corpus size).

Known limitation, disclosed rather than hidden: this is single-process
state. Running the API behind multiple worker processes or replicas means
each one enforces its own independent limit - the effective limit is
(configured limit) x (number of processes), not the configured limit
itself. Fine for a single-process deployment; replace with a shared store
(Redis, etc.) before scaling out to multiple workers/replicas.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """True and records the request if ``key`` is under its limit
        within the current window; False (and does not record) if not."""
        now = time.time()
        with self._lock:
            timestamps = self._requests[key]
            while timestamps and now - timestamps[0] > self.window_seconds:
                timestamps.popleft()
            if len(timestamps) >= self.max_requests:
                return False
            timestamps.append(now)
            return True
