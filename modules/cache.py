import os
import time
import json
import hashlib
import threading
from typing import Any, Dict, Optional, Callable
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

class CacheManager:
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._ttl = ttl
        self._max_size = max_size
        self._access_times: Dict[str, float] = {}

    def _generate_key(self, func_name: str, **kwargs) -> str:
        key_data = f"{func_name}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry["timestamp"] < self._ttl:
                    self._access_times[key] = time.time()
                    return entry["data"]
                else:
                    del self._cache[key]
                    del self._access_times[key]
            return None

    def set(self, key: str, value: Any) -> None:
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
