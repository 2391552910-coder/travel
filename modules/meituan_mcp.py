import os
import time
import requests
import hashlib
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from modules.cache import get_cache
from modules.error_handler import with_retry, APIError, format_api_error

load_dotenv()

class MeituanMCPService:
    def __init__(self):
        self.api_key = os.getenv("MEITUAN_API_KEY")
        self.app_id = os.getenv("MEITUAN_APP_ID")
        self.base_url = os.getenv("MEITUAN_BASE_URL", "https://api.meituan.com")
        self.cache = get_cache()

    def _generate_sign(self, params: Dict[str, Any]) -> str:
        sorted_params = sorted(params.items())
        sign_str = "".join(f"{k}{v}" for k, v in sorted_params)
        sign_str += self.api_key
        return hashlib.md5(sign_str.encode()).hexdigest()

    @with_retry("Meituan")
    def search_restaurants(self, keywords: str, city: str,
                          category: Optional[str] = None,
                          limit: int = 20) -> Dict[str, Any]:
        cache_key = f"meituan_food:{keywords}:{city}:{category}:{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        url = f"{self.base_url}/search/restaurants"
        params = {
            "appId": self.app_id,
            "keyword": keywords,
            "city": city,
            "limit": limit,
            "timestamp": int(time.time() * 1000)
        }
        if category:
            params["category"] = category

        params["sign"] = self._generate_sign(params)

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                raise format_api_error("Meituan", response.status_code, None)

            data = response.json()

            if data.get("code") != 200:
                raise APIError("Meituan", data.get("msg", "Search failed"), data.get("code"))

            result = {
                "restaurants": data.get("data", []),
                "total": data.get("total", 0),
                "page": data.get("page", 1)
            }

            self.cache.set(cache_key, result)
            return result
        except APIError:
            raise
        except Exception as e:
            raise APIError("Meituan", f"请求失败: {str(e)}", None)

    @with_retry("Meituan")
    def search_hotels(self, city: str, check_in: str, check_out: str,
                      keywords: Optional[str] = None,
                      star_level: Optional[int] = None,
                      price_range: Optional[tuple] = None,
                      limit: int = 20) -> Dict[str, Any]:
        cache_key = f"meituan_hotel:{city}:{check_in}:{check_out}:{keywords}:{star_level}:{price_range}:{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        url = f"{self.base_url}/search/hotels"
        params = {
            "appId": self.app_id,
            "city": city,
            "checkIn": check_in,
            "checkOut": check_out,
            "limit": limit,
            "timestamp": int(time.time() * 1000)
        }
        if keywords:
            params["keyword"] = keywords
        if star_level:
            params["star"] = star_level
        if price_range:
            params["minPrice"], params["maxPrice"] = price_range
        
        params["sign"] = self._generate_sign(params)
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            raise format_api_error("Meituan", response.status_code, None)
        
        data = response.json()
        
        if data.get("code") != 200:
            raise APIError("Meituan", data.get("msg", "Search failed"), data.get("code"))
        
        result = {
            "hotels": data.get("data", []),
            "total": data.get("total", 0),
            "filters": data.get("filters", {})
        }
        
        self.cache.set(cache_key, result)
        return result

    @with_retry("Meituan")
    def get_restaurant_detail(self, poi_id: str) -> Dict[str, Any]:
        cache_key = f"meituan_restaurant_detail:{poi_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        url = f"{self.base_url}/restaurant/{poi_id}"
        params = {
            "appId": self.app_id,
            "timestamp": int(time.time() * 1000)
        }
        params["sign"] = self._generate_sign(params)
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            raise format_api_error("Meituan", response.status_code, None)
        
        data = response.json()
        
        if data.get("code") != 200:
            raise APIError("Meituan", data.get("msg", "Failed to get detail"), data.get("code"))
        
        result = data.get("data", {})
        
        self.cache.set(cache_key, result)
        return result

    @with_retry("Meituan")
    def get_hotel_detail(self, poi_id: str) -> Dict[str, Any]:
        cache_key = f"meituan_hotel_detail:{poi_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        url = f"{self.base_url}/hotel/{poi_id}"
        params = {
            "appId": self.app_id,
            "timestamp": int(time.time() * 1000)
        }
        params["sign"] = self._generate_sign(params)
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            raise format_api_error("Meituan", response.status_code, None)
        
        data = response.json()
        
        if data.get("code") != 200:
            raise APIError("Meituan", data.get("msg", "Failed to get detail"), data.get("code"))
        
        result = data.get("data", {})
        
        self.cache.set(cache_key, result)
        return result

    @with_retry("Meituan")
    def get_nearby_restaurants(self, latitude: float, longitude: float,
                                radius: int = 3000, limit: int = 20) -> Dict[str, Any]:
        cache_key = f"meituan_nearby_food:{latitude}:{longitude}:{radius}:{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        url = f"{self.base_url}/lbs/nearby/restaurants"
        params = {
            "appId": self.app_id,
            "latitude": latitude,
            "longitude": longitude,
            "radius": radius,
            "limit": limit,
            "timestamp": int(time.time() * 1000)
        }
        params["sign"] = self._generate_sign(params)
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            raise format_api_error("Meituan", response.status_code, None)
        
        data = response.json()
        
        if data.get("code") != 200:
            raise APIError("Meituan", data.get("msg", "Search failed"), data.get("code"))
        
        result = {
            "restaurants": data.get("data", []),
            "total": data.get("total", 0)
        }
        
        self.cache.set(cache_key, result)
        return result

    def format_restaurant_info(self, restaurant: Dict[str, Any]) -> str:
        return f"""
餐厅名称: {restaurant.get('name', 'N/A')}
地址: {restaurant.get('address', 'N/A')}
评分: {restaurant.get('rating', 'N/A')}分
人均价格: {restaurant.get('avgPrice', 'N/A')}元
菜系: {restaurant.get('category', 'N/A')}
距您: {restaurant.get('distance', 'N/A')}米
"""

    def format_hotel_info(self, hotel: Dict[str, Any]) -> str:
        return f"""
酒店名称: {hotel.get('name', 'N/A')}
地址: {hotel.get('address', 'N/A')}
星级: {hotel.get('star', 'N/A')}星
价格: {hotel.get('price', 'N/A')}元/晚
评分: {hotel.get('rating', 'N/A')}分
标签: {', '.join(hotel.get('tags', []))}
"""

_meituan_service: Optional[MeituanMCPService] = None

def get_meituan_service() -> MeituanMCPService:
    global _meituan_service
    if _meituan_service is None:
        _meituan_service = MeituanMCPService()
    return _meituan_service
