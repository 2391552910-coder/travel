import unittest
import time
import os
import sys
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from modules.cache import CacheManager, get_cache
from modules.error_handler import (
    ErrorHandler, APIError, RateLimitError,
    DataNotFoundError, with_retry, validate_response
)
from modules.amap_mcp import AmapMCPService
from modules.meituan_mcp import MeituanMCPService
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

class TestCacheManager(unittest.TestCase):
    def setUp(self):
        self.cache = CacheManager(ttl=2, max_size=3)

    def test_cache_set_and_get(self):
        self.cache.set("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")

    def test_cache_expiration(self):
        self.cache.set("key1", "value1")
        time.sleep(2.5)
        self.assertIsNone(self.cache.get("key1"))

    def test_cache_lru_eviction(self):
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")
        self.cache.set("key4", "value4")
        self.assertIsNone(self.cache.get("key1"))
        self.assertIsNotNone(self.cache.get("key4"))

    def test_generate_key_consistency(self):
        key1 = self.cache._generate_key("func", arg1="a", arg2="b")
        key2 = self.cache._generate_key("func", arg2="b", arg1="a")
        self.assertEqual(key1, key2)

class TestErrorHandler(unittest.TestCase):
    def setUp(self):
        self.handler = ErrorHandler(max_retries=2, base_delay=0.1)

    def test_exponential_backoff(self):
        self.assertAlmostEqual(self.handler.exponential_backoff(0), 0.1, places=1)
        self.assertAlmostEqual(self.handler.exponential_backoff(1), 0.2, places=1)

    def test_should_retry_rate_limit(self):
        self.assertTrue(self.handler.should_retry(RateLimitError("test", "rate limit", 429)))

    def test_should_not_retry_data_not_found(self):
        self.assertFalse(self.handler.should_retry(DataNotFoundError("test", "not found", 404)))

class TestValidateResponse(unittest.TestCase):
    def test_valid_response(self):
        response = {"field1": "value1", "field2": "value2"}
        self.assertTrue(validate_response(response, ["field1", "field2"]))

    def test_missing_field(self):
        response = {"field1": "value1"}
        self.assertFalse(validate_response(response, ["field1", "field2"]))

    def test_invalid_type(self):
        self.assertFalse(validate_response("not a dict", ["field1"]))

class TestAmapMCPService(unittest.TestCase):
    def setUp(self):
        self.amap = AmapMCPService()
        self.mock_response_data = {
            "status": "1",
            "info": "OK",
            "count": "1",
            "pois": [{
                "name": "故宫",
                "address": "北京市东城区景山前街4号",
                "type": "风景名胜",
                "location": "116.397026,39.918058",
                "distance": "0",
                "tel": "010-85007921"
            }]
        }

    @patch('modules.amap_mcp.requests.get')
    def test_search_place_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_response_data
        mock_get.return_value = mock_response

        result = self.amap.search_place("故宫", city="北京")
        
        self.assertEqual(result["count"], "1")
        self.assertEqual(len(result["pois"]), 1)
        self.assertEqual(result["pois"][0]["name"], "故宫")

    @patch('modules.amap_mcp.requests.get')
    def test_search_place_api_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "0", "info": "INVALID_USER_KEY"}
        mock_get.return_value = mock_response

        from modules.error_handler import APIError
        with self.assertRaises(APIError):
            self.amap.search_place("test")

    def test_format_place_info(self):
        poi = self.mock_response_data["pois"][0]
        formatted = self.amap.format_place_info(poi)
        
        self.assertIn("故宫", formatted)
        self.assertIn("北京市东城区景山前街4号", formatted)
        self.assertIn("116.397026,39.918058", formatted)

class TestMeituanMCPService(unittest.TestCase):
    def setUp(self):
        self.meituan = MeituanMCPService()
        self.mock_restaurant_data = {
            "code": 200,
            "msg": "success",
            "data": [{
                "name": "全聚德烤鸭店",
                "address": "北京市东城区前门大街30号",
                "rating": "4.5",
                "avgPrice": 200,
                "category": "烤鸭",
                "distance": 500
            }],
            "total": 1
        }

    @patch('modules.meituan_mcp.requests.get')
    def test_search_restaurants_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_restaurant_data
        mock_get.return_value = mock_response

        result = self.meituan.search_restaurants("烤鸭", "北京")
        
        self.assertEqual(result["total"], 1)
        self.assertEqual(len(result["restaurants"]), 1)
        self.assertEqual(result["restaurants"][0]["name"], "全聚德烤鸭店")

    def test_format_restaurant_info(self):
        restaurant = self.mock_restaurant_data["data"][0]
        formatted = self.meituan.format_restaurant_info(restaurant)
        
        self.assertIn("全聚德烤鸭店", formatted)
        self.assertIn("4.5", formatted)
        self.assertIn("200", formatted)

    def test_generate_sign_consistency(self):
        params1 = {"a": "1", "b": "2"}
        params2 = {"b": "2", "a": "1"}
        sign1 = self.meituan._generate_sign(params1)
        sign2 = self.meituan._generate_sign(params2)
        self.assertEqual(sign1, sign2)

class TestRetryDecorator(unittest.TestCase):
    @patch('time.sleep')
    def test_retry_success_on_third_attempt(self, mock_sleep):
        call_count = [0]
        
        @with_retry("TestService")
        def flaky_function():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RateLimitError("TestService", "rate limit", 429)
            return "success"
        
        result = flaky_function()
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch('time.sleep')
    def test_no_retry_on_data_not_found(self, mock_sleep):
        @with_retry("TestService")
        def not_found_function():
            raise DataNotFoundError("TestService", "not found", 404)
        
        with self.assertRaises(DataNotFoundError):
            not_found_function()
        
        mock_sleep.assert_not_called()

class TestConcurrency(unittest.TestCase):
    def test_cache_thread_safety(self):
        import threading
        cache = CacheManager(ttl=10, max_size=100)
        results = []
        
        def worker(n):
            key = f"key_{n % 10}"
            cache.set(key, n)
            val = cache.get(key)
            results.append(val)
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(results), 50)

if __name__ == "__main__":
    unittest.main(verbosity=2)
