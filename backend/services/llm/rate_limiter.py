import asyncio
import time
from collections import deque


class RateLimiter:
    """Simple sliding-window limiter for outbound API calls."""

    def __init__(self, max_requests: int = 9, per_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        while True:
            sleep_for = 0.0
            async with self._lock:
                now = time.monotonic()

                while self._timestamps and now - self._timestamps[0] > self.per_seconds:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(now)
                    return

                sleep_for = self.per_seconds - (now - self._timestamps[0]) + 0.05

            await asyncio.sleep(max(sleep_for, 0.05))


# Gemini free tier: 10 RPM — leave 1 RPM headroom
gemini_limiter = RateLimiter(max_requests=9, per_seconds=60.0)

# Groq free tier: 30 RPM — leave 2 RPM headroom for retries
groq_limiter = RateLimiter(max_requests=28, per_seconds=60.0)
