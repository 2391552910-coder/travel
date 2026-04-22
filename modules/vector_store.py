import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

load_dotenv()

class VectorStore:
    def __init__(self, persist_directory: str = "./data/vector_store"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        self.attractions_collection = self.client.get_or_create_collection(
            name="attractions",
            metadata={"description": "景点知识库"}
        )

        self.preferences_collection = self.client.get_or_create_collection(
            name="user_preferences",
            metadata={"description": "用户偏好历史"}
        )

    def add_attraction(self, attraction_data: Dict[str, Any]) -> str:
        doc_id = f"attr_{attraction_data.get('name', '')}_{datetime.now().timestamp()}"

        content = self._format_attraction_content(attraction_data)

        self.attractions_collection.add(
            documents=[content],
            ids=[doc_id],
            metadatas=[{
                "name": attraction_data.get("name", ""),
                "city": attraction_data.get("city", ""),
                "type": attraction_data.get("type", ""),
                "rating": attraction_data.get("rating", 0),
                "added_at": datetime.now().isoformat()
            }]
        )

        return doc_id

    def _format_attraction_content(self, data: Dict[str, Any]) -> str:
        parts = [
            f"景点名称: {data.get('name', 'N/A')}",
            f"城市: {data.get('city', 'N/A')}",
            f"地址: {data.get('address', 'N/A')}",
            f"类型: {data.get('type', 'N/A')}",
            f"评分: {data.get('rating', 'N/A')}",
            f"开放时间: {data.get('open_hours', 'N/A')}",
            f"门票: {data.get('ticket', 'N/A')}",
            f"简介: {data.get('description', 'N/A')}",
            f"特色: {data.get('highlights', 'N/A')}",
        ]
        return " | ".join([p for p in parts if p])

    def search_attractions(
        self,
        query: str,
        city: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        where_filter = {"city": city} if city else None

        try:
            results = self.attractions_collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_filter
            )

            return self._format_search_results(results)
        except Exception:
            return []

    def _format_search_results(self, results) -> List[Dict[str, Any]]:
        formatted = []
        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"]):
                metadata = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
                formatted.append({
                    "content": doc,
                    "name": metadata.get("name", ""),
                    "city": metadata.get("city", ""),
                    "type": metadata.get("type", ""),
                    "rating": metadata.get("rating", 0)
                })
        return formatted

    def add_user_preference(
        self,
        user_id: str,
        destination: str,
        days: int,
        budget: int,
        preferences: List[str],
        selected_places: List[Dict[str, Any]]
    ) -> str:
        doc_id = f"pref_{user_id}_{datetime.now().timestamp()}"

        content = self._format_preference_content(
            destination, days, budget, preferences, selected_places
        )

        self.preferences_collection.add(
            documents=[content],
            ids=[doc_id],
            metadatas=[{
                "user_id": user_id,
                "destination": destination,
                "days": days,
                "budget": budget,
                "preferences": json.dumps(preferences),
                "added_at": datetime.now().isoformat()
            }]
        )

        return doc_id

    def _format_preference_content(
        self,
        destination: str,
        days: int,
        budget: int,
        preferences: List[str],
        selected_places: List[Dict[str, Any]]
    ) -> str:
        places_str = json.dumps(selected_places, ensure_ascii=False)
        return (
            f"目的地: {destination} | "
            f"天数: {days}天 | "
            f"预算: {budget}元 | "
            f"偏好: {', '.join(preferences)} | "
            f"选择景点: {places_str}"
        )

    def get_user_preferences(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            results = self.preferences_collection.query(
                query_texts=[user_id],
                n_results=limit,
                where={"user_id": user_id}
            )

            return self._format_preference_results(results)
        except Exception:
            return []

    def _format_preference_results(self, results) -> List[Dict[str, Any]]:
        formatted = []
        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"]):
                metadata = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
                formatted.append({
                    "content": doc,
                    "destination": metadata.get("destination", ""),
                    "days": metadata.get("days", 0),
                    "budget": metadata.get("budget", 0),
                    "preferences": json.loads(metadata.get("preferences", "[]")),
                    "added_at": metadata.get("added_at", "")
                })
        return formatted

    def retrieve_relevant_context(
        self,
        query: str,
        user_id: Optional[str] = None,
        city: Optional[str] = None
    ) -> str:
        context_parts = []

        attractions = self.search_attractions(query=query, city=city, limit=3)
        if attractions:
            context_parts.append("【相关景点知识】")
            for attr in attractions:
                context_parts.append(attr["content"])

        if user_id:
            prefs = self.get_user_preferences(user_id, limit=2)
            if prefs:
                context_parts.append("\n【用户历史偏好】")
                for pref in prefs:
                    context_parts.append(pref["content"])

        return "\n".join(context_parts) if context_parts else ""

_vector_store: Optional[VectorStore] = None

def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        persist_dir = os.getenv("VECTOR_STORE_DIR", "./data/vector_store")
        _vector_store = VectorStore(persist_directory=persist_dir)
    return _vector_store
