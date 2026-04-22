import os
import time
import json
import hashlib
import threading
from typing import Any, Dict, Optional, Callable
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

CACHE_TTL_CONFIG = {
    "place": 86400,
    "restaurant": 3600,
    "hotel": 3600,
    "route": 1800,
    "geocode": 604800,
}

def get_cache_ttl(category: str) -> int:
    return CACHE_TTL_CONFIG.get(category, 3600)

class CacheManager:
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._default_ttl = ttl
        self._max_size = max_size
        self._access_times: Dict[str, float] = {}
        self._category_ttls: Dict[str, int] = {}

    def _generate_key(self, func_name: str, **kwargs) -> str:
        key_data = f"{func_name}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_ttl(self, category: Optional[str] = None) -> int:
        if category and category in CACHE_TTL_CONFIG:
            return CACHE_TTL_CONFIG[category]
        return self._default_ttl

    def get(self, key: str, category: Optional[str] = None) -> Optional[Any]:
        ttl = self._get_ttl(category)
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry["timestamp"] < ttl:
                    self._access_times[key] = time.time()
                    return entry["data"]
                else:
                    del self._cache[key]
                    if key in self._access_times:
                        del self._access_times[key]
            return None

    def set(self, key: str, value: Any, category: Optional[str] = None) -> None:
        with self._lock:
            if len(self._cache) >= self._max_size:
                self._evict_lru()
            self._cache[key] = {
                "data": value,
                "timestamp": time.time()
            }
            self._access_times[key] = time.time()

    def _evict_lru(self) -> None:
        if not self._access_times:
            return
        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        del self._cache[lru_key]
        del self._access_times[lru_key]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._access_times.clear()

    def cached(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = self._generate_key(func.__name__, args=args, **kwargs)
            cached_value = self.get(cache_key)
            if cached_value is not None:
                return cached_value
            result = func(*args, **kwargs)
            self.set(cache_key, result)
            return result
        return wrapper

_cache_manager = CacheManager(
    ttl=int(os.getenv("CACHE_TTL", 3600)),
    max_size=int(os.getenv("CACHE_MAX_SIZE", 1000))
)

def get_cache() -> CacheManager:
    return _cache_manager
