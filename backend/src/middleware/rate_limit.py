"""
Rate limiting middleware for API protection.
Implements rate limiting using in-memory storage or Redis for distributed rate limiting.
"""
import time
import logging
from typing import Dict, Optional
from fastapi import Request, HTTPException, status
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests: int  # Number of requests allowed
    window: int    # Time window in seconds


class InMemoryRateLimiter:
    """In-memory rate limiter implementation."""

    def __init__(self):
        # Dictionary to store request counts per identifier
        # Key: identifier (IP, user_id, etc.), Value: deque of timestamps
        self.requests: Dict[str, deque] = defaultdict(deque)

    def is_allowed(self, identifier: str, limit: RateLimit) -> bool:
        """
        Check if a request is allowed based on rate limit.

        Args:
            identifier: Unique identifier for the requester (e.g., IP address)
            limit: Rate limit configuration

        Returns:
            True if request is allowed, False otherwise
        """
        current_time = time.time()
        window_start = current_time - limit.window

        # Remove old requests outside the time window
        while self.requests[identifier] and self.requests[identifier][0] < window_start:
            self.requests[identifier].popleft()

        # Check if we're under the limit
        if len(self.requests[identifier]) < limit.requests:
            # Add current request
            self.requests[identifier].append(current_time)
            return True

        return False

    def get_reset_time(self, identifier: str, limit: RateLimit) -> float:
        """
        Get the time when the rate limit will reset.

        Args:
            identifier: Unique identifier for the requester
            limit: Rate limit configuration

        Returns:
            Unix timestamp when the rate limit will reset
        """
        if identifier in self.requests and self.requests[identifier]:
            oldest_request = self.requests[identifier][0]
            return oldest_request + limit.window
        return time.time()


# Default rate limits
DEFAULT_LIMITS = {
    'global': RateLimit(requests=100, window=3600),  # 100 requests per hour
    'user': RateLimit(requests=10, window=60),      # 10 requests per minute
    'burst': RateLimit(requests=5, window=10)       # 5 requests per 10 seconds
}


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting."""

    def __init__(
        self,
        rate_limiter: InMemoryRateLimiter = None,
        default_limit: str = 'user',
        global_limit: str = 'global',
        burst_limit: str = 'burst'
    ):
        self.rate_limiter = rate_limiter or InMemoryRateLimiter()
        self.default_limit = DEFAULT_LIMITS[default_limit]
        self.global_limit = DEFAULT_LIMITS[global_limit]
        self.burst_limit = DEFAULT_LIMITS[burst_limit]

    async def __call__(self, request: Request, call_next):
        """
        Middleware to check rate limits before processing the request.
        """
        # Get client IP address
        client_ip = self.get_client_ip(request)

        # Check burst limit (most restrictive, shortest window)
        if not self.rate_limiter.is_allowed(f"burst:{client_ip}", self.burst_limit):
            reset_time = self.rate_limiter.get_reset_time(f"burst:{client_ip}", self.burst_limit)
            return self.rate_limit_response(reset_time, "Too many requests in short time")

        # Check default limit
        if not self.rate_limiter.is_allowed(f"default:{client_ip}", self.default_limit):
            reset_time = self.rate_limiter.get_reset_time(f"default:{client_ip}", self.default_limit)
            return self.rate_limit_response(reset_time, "Too many requests")

        # Check global limit (longest window)
        if not self.rate_limiter.is_allowed(f"global:{client_ip}", self.global_limit):
            reset_time = self.rate_limiter.get_reset_time(f"global:{client_ip}", self.global_limit)
            return self.rate_limit_response(reset_time, "Too many requests today")

        # If all checks pass, process the request
        response = await call_next(request)

        # Add rate limit headers to response
        self.add_rate_limit_headers(response, client_ip)

        return response

    def get_client_ip(self, request: Request) -> str:
        """
        Extract client IP address from request.
        """
        # Check for X-Forwarded-For header (common with reverse proxies/load balancers)
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # Take the first IP if multiple are provided
            return forwarded.split(",")[0].strip()

        # Check for X-Real-IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        # Fallback to client host
        return request.client.host if request.client else "unknown"

    async def rate_limit_response(self, reset_time: float, message: str = "Rate limit exceeded"):
        """
        Create a rate limit exceeded response.
        """
        from starlette.responses import JSONResponse

        current_time = time.time()
        retry_after = int(reset_time - current_time) + 1  # Add 1 second buffer

        response_headers = {
            "X-RateLimit-Limit": str(self.default_limit.requests),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(reset_time)),
            "Retry-After": str(retry_after)
        }

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "message": message,
                "retry_after": retry_after
            },
            headers=response_headers
        )

    def add_rate_limit_headers(self, response, client_ip: str):
        """
        Add rate limit information to response headers.
        """
        remaining_default = max(
            0,
            self.default_limit.requests - len(self.rate_limiter.requests[f"default:{client_ip}"])
        )

        response.headers["X-RateLimit-Limit"] = str(self.default_limit.requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining_default)

        if self.rate_limiter.requests[f"default:{client_ip}"]:
            reset_time = self.rate_limiter.requests[f"default:{client_ip}"][0] + self.default_limit.window
            response.headers["X-RateLimit-Reset"] = str(int(reset_time))


# Alternative Redis-based rate limiter (requires redis-py)
class RedisRateLimiter:
    """
    Redis-based rate limiter for distributed applications.
    Uncomment and use if Redis is available.
    """
    # def __init__(self, redis_client, default_prefix: str = "rate_limit:"):
    #     self.redis = redis_client
    #     self.prefix = default_prefix
    #
    # def is_allowed(self, identifier: str, limit: RateLimit) -> bool:
    #     key = f"{self.prefix}{identifier}"
    #     current_time = time.time()
    #     window_start = current_time - limit.window
    #
    #     # Remove old entries
    #     self.redis.zremrangebyscore(key, 0, window_start)
    #
    #     # Get current count
    #     current_requests = self.redis.zcard(key)
    #
    #     if current_requests < limit.requests:
    #         # Add current request
    #         self.redis.zadd(key, {str(current_time): current_time})
    #         # Set expiration
    #         self.redis.expire(key, limit.window)
    #         return True
    #
    #     return False


# Convenience function to create rate limiting middleware
def create_rate_limit_middleware(
    default_limit: str = 'user',
    global_limit: str = 'global',
    burst_limit: str = 'burst'
) -> RateLimitMiddleware:
    """
    Create a rate limiting middleware with specified limits.

    Args:
        default_limit: Default rate limit ('user', 'global', 'burst')
        global_limit: Global rate limit
        burst_limit: Burst rate limit

    Returns:
        Configured RateLimitMiddleware instance
    """
    return RateLimitMiddleware(
        default_limit=default_limit,
        global_limit=global_limit,
        burst_limit=burst_limit
    )