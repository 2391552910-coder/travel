import os
import time
import logging
from typing import Any, Callable, Dict, Optional, TypeVar
from functools import wraps
from dotenv import load_dotenv
from enum import Enum

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')

class APIError(Exception):
    def __init__(self, service: str, message: str, status_code: Optional[int] = None):
        self.service = service
        self.message = message
        self.status_code = status_code
        super().__init__(f"[{service}] {message}")

class RateLimitError(APIError):
    pass

class DataNotFoundError(APIError):
    pass

class ServiceUnavailableError(APIError):
    pass

class ErrorHandler:
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def exponential_backoff(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        return delay

    def should_retry(self, exception: Exception) -> bool:
        if isinstance(exception, RateLimitError):
            return True
        if isinstance(exception, ServiceUnavailableError):
            return True
        if isinstance(exception, APIError) and "timeout" in str(exception).lower():
            return True
        return False

    def handle_error(self, service: str, exception: Exception, attempt: int) -> None:
        error_msg = f"[{service}] Attempt {attempt + 1} failed: {str(exception)}"
        if self.should_retry(exception):
            delay = self.exponential_backoff(attempt)
            logger.warning(f"{error_msg}. Retrying in {delay:.2f}s...")
            time.sleep(delay)
        else:
            logger.error(f"{error_msg}. No retry.")
            raise exception

def with_retry(service_name: str = "Unknown"):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        handler = ErrorHandler()
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(handler.max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    handler.handle_error(service_name, e, attempt)
            raise last_exception
        return wrapper
    return decorator

def validate_response(response: Dict[str, Any], required_fields: list) -> bool:
    if not isinstance(response, dict):
        return False
    for field in required_fields:
        if field not in response:
            logger.warning(f"Missing required field: {field}")
            return False
    return True

def format_api_error(service: str, status_code: Optional[int], response_body: Optional[Dict]) -> APIError:
    if status_code == 404:
        return DataNotFoundError(service, "Resource not found", status_code)
    elif status_code == 429:
        return RateLimitError(service, "Rate limit exceeded", status_code)
    elif status_code >= 500:
        return ServiceUnavailableError(service, "Service unavailable", status_code)
    else:
        error_msg = response_body.get("message", "Unknown error") if response_body else "Unknown error"
        return APIError(service, error_msg, status_code)
