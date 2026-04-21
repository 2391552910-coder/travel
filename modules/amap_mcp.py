import os
import requests
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from modules.cache import get_cache
from modules.error_handler import with_retry, APIError, format_api_error

load_dotenv()

class AmapMCPService:
    def __init__(self):
        self.api_key = os.getenv("AMAP_API_KEY")
        self.base_url = os.getenv("AMAP_BASE_URL", "https://restapi.amap.com/v3")
        self.cache = get_cache()

    @with_retry("Amap")
    def search_place(self, keywords: str, city: Optional[str] = None, 
                     types: Optional[str] = None, offset: int = 20, 
                     page: int = 1) -> Dict[str, Any]:
        cache_key = f"amap_search:{keywords}:{city}:{types}:{offset}:{page}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        url = f"{self.base_url}/place/text"
        params = {
            "key": self.api_key,
            "keywords": keywords,
            "offset": offset,
            "page": page,
            "output": "json"
        }
        if city:
            params["city"] = city
        if types:
            params["types"] = types

        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            raise format_api_error("Amap", response.status_code, None)
        
        data = response.json()
        
        if data.get("status") != "1":
            raise APIError("Amap", data.get("info", "Search failed"), None)
        
        result = {
            "pois": data.get("pois", []),
            "count": data.get("count", "0"),
            "info": data.get("info", ""),
            "suggested_keyword": data.get("suggested_keyword", "")
        }
        
        self.cache.set(cache_key, result)
        return result

    @with_retry("Amap")
    def get_around_places(self, location: str, keywords: Optional[str] = None,
                         types: Optional[str] = None, radius: int = 1000) -> Dict[str, Any]:
        cache_key = f"amap_around:{location}:{keywords}:{types}:{radius}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        url = f"{self.base_url}/place/around"
        params = {
            "key": self.api_key,
            "location": location,
            "radius": radius,
            "output": "json"
        }
        if keywords:
            params["keywords"] = keywords
        if types:
            params["types"] = types

        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            raise format_api_error("Amap", response.status_code, None)
        
        data = response.json()
        
        if data.get("status") != "1":
            raise APIError("Amap", data.get("info", "Search failed"), None)
        
        result = {
            "pois": data.get("pois", []),
            "bounds": data.get("bounds", ""),
            "recommend": data.get("recommend", "")
        }
        
        self.cache.set(cache_key, result)
        return result

    @with_retry("Amap")
    def get_route_directions(self, origin: str, destination: str,
                            strategy: str = "0", mode: str = "walking") -> Dict[str, Any]:
        cache_key = f"amap_route:{origin}:{destination}:{strategy}:{mode}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        url = f"{self.base_url}/direction/{mode}"
        params = {
            "key": self.api_key,
            "origin": origin,
            "destination": destination,
            "strategy": strategy,
            "output": "json"
        }

        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            raise format_api_error("Amap", response.status_code, None)
        
        data = response.json()
        
        if data.get("status") != "1":
            raise APIError("Amap", data.get("info", "Route planning failed"), None)
        
        result = {
            "route": data.get("route", {}),
            "info": data.get("info", ""),
            "count": data.get("count", "0")
        }
        
        self.cache.set(cache_key, result)
        return result

    @with_retry("Amap")
    def get_geocode(self, address: str, city: Optional[str] = None) -> Dict[str, Any]:
        cache_key = f"amap_geocode:{address}:{city}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        url = f"{self.base_url}/geocode/geo"
        params = {
            "key": self.api_key,
            "address": address,
            "output": "json"
        }
        if city:
            params["city"] = city

        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            raise format_api_error("Amap", response.status_code, None)
        
        data = response.json()
        
        if data.get("status") != "1":
            raise APIError("Amap", data.get("info", "Geocoding failed"), None)
        
        result = {
            "geocodes": data.get("geocodes", []),
            "info": data.get("info", ""),
            "count": data.get("count", "0")
        }
        
        self.cache.set(cache_key, result)
        return result

    def format_place_info(self, poi: Dict[str, Any]) -> str:
        return f"""
名称: {poi.get('name', 'N/A')}
地址: {poi.get('address', 'N/A')}
类型: {poi.get('type', 'N/A')}
距离: {poi.get('distance', 'N/A')}米
电话: {poi.get('tel', 'N/A')}
经纬度: {poi.get('location', 'N/A')}
"""

    def format_route_info(self, route_data: Dict[str, Any]) -> str:
        result = []
        paths = route_data.get("route", {}).get("paths", [])
        
        for i, path in enumerate(paths, 1):
            result.append(f"\n路线 {i}:")
            result.append(f"距离: {path.get('distance', 'N/A')}米")
            result.append(f"预计时间: {path.get('duration', 'N/A')}秒")
            
            steps = path.get('steps', [])
            if steps:
                result.append("途经地点:")
                for j, step in enumerate(steps[:5], 1):
                    result.append(f"  {j}. {step.get('instruction', '')}")
        
        return "\n".join(result)

_amap_service: Optional[AmapMCPService] = None

def get_amap_service() -> AmapMCPService:
    global _amap_service
    if _amap_service is None:
        _amap_service = AmapMCPService()
    return _amap_service
