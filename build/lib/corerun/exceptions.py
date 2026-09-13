"""
corerun SDK exceptions.
"""


class CoreRunError(Exception):
    """Base exception for corerun SDK"""
    pass


class AuthenticationError(CoreRunError):
    """Authentication failed (401)"""
    pass


class NotFoundError(CoreRunError):
    """Resource not found (404)"""
    pass


class ValidationError(CoreRunError):
    """Request validation failed (400, 422)"""
    pass


class RateLimitError(CoreRunError):
    """Rate limit exceeded (429)"""
    pass


class ServerError(CoreRunError):
    """Server error (5xx)"""
    pass


class TimeoutError(CoreRunError):
    """Request timed out"""
    pass


class ConfigurationError(CoreRunError):
    """Configuration error"""
    pass
