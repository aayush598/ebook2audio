"""
Rate Limiter module — extracted from the original single-file script.
No changes to functionality or logic have been made.
"""

from datetime import datetime
from typing import Tuple
import time


class RateLimiter:
    """Manages API rate limits"""
    
    def __init__(self, rpm: int, tpm: int, rpd: int):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.request_times = []
        self.daily_requests = 0
        self.last_reset = datetime.now()
    
    def can_make_request(self) -> Tuple[bool, str]:
        now = datetime.now()
        if (now - self.last_reset).days >= 1:
            self.daily_requests = 0
            self.last_reset = now
        if self.daily_requests >= self.rpd:
            return False, f"Daily limit reached ({self.rpd} requests/day)"
        self.request_times = [t for t in self.request_times if (now - t).seconds < 60]
        if len(self.request_times) >= self.rpm:
            wait_time = 60 - (now - self.request_times[0]).seconds
            return False, f"Rate limit: wait {wait_time}s (max {self.rpm} requests/min)"
        return True, "OK"
    
    def record_request(self):
        self.request_times.append(datetime.now())
        self.daily_requests += 1
    
    def get_wait_time(self) -> int:
        if not self.request_times:
            return 0
        now = datetime.now()
        oldest = self.request_times[0]
        elapsed = (now - oldest).seconds
        if elapsed < 60:
            return max(0, 60 - elapsed + 1)
        return 0
