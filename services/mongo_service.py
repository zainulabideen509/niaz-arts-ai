"""
MongoDB Service - Logs every AI recommendation and provides analytics.
Stores data in MongoDB Atlas (free tier).
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

logger = logging.getLogger("niaz-arts-ai.mongo")

MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME = "niazarts"
COLLECTION_LOGS = "recommendation_logs"


class MongoService:
    def __init__(self):
        self.client = None
        self.db = None
        self.logs = None

        if not MONGO_URI:
            logger.warning("MONGO_URI not set! Logging disabled.")
            return

        try:
            self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            self.client.admin.command("ping")
            self.db = self.client[DB_NAME]
            self.logs = self.db[COLLECTION_LOGS]
            logger.info("Connected to MongoDB Atlas successfully.")
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            self.client = None

    def is_connected(self) -> bool:
        return self.client is not None and self.logs is not None

    # ── LOGGING ──────────────────────────────────────────────

    def log_image_recommendation(
        self,
        wall_colors: list,
        results: list,
        platform: str = "unknown",
    ):
        """Log an image-based recommendation request."""
        if not self.is_connected():
            return

        try:
            doc = {
                "type": "image",
                "wall_colors": wall_colors,
                "results_count": len(results),
                "top_results": [
                    {"title": r["title"], "score": r["score"], "product_id": r["product_id"]}
                    for r in results[:6]
                ],
                "platform": platform,
                "timestamp": datetime.utcnow(),
            }
            self.logs.insert_one(doc)
            logger.info("Logged image recommendation.")
        except Exception as e:
            logger.error(f"Failed to log: {e}")

    def log_text_recommendation(
        self,
        query: str,
        results: list,
        platform: str = "unknown",
    ):
        """Log a text-based recommendation request."""
        if not self.is_connected():
            return

        try:
            doc = {
                "type": "text",
                "query": query,
                "results_count": len(results),
                "top_results": [
                    {"title": r["title"], "score": r["score"], "product_id": r["product_id"]}
                    for r in results[:6]
                ],
                "platform": platform,
                "timestamp": datetime.utcnow(),
            }
            self.logs.insert_one(doc)
            logger.info(f"Logged text recommendation: '{query}'")
        except Exception as e:
            logger.error(f"Failed to log: {e}")

    # ── ANALYTICS ────────────────────────────────────────────

    def get_total_recommendations(self, days: int = 30) -> dict:
        """Get total recommendation counts."""
        if not self.is_connected():
            return {"total": 0, "image": 0, "text": 0}

        since = datetime.utcnow() - timedelta(days=days)
        total = self.logs.count_documents({"timestamp": {"$gte": since}})
        image = self.logs.count_documents({"type": "image", "timestamp": {"$gte": since}})
        text = self.logs.count_documents({"type": "text", "timestamp": {"$gte": since}})

        return {"total": total, "image": image, "text": text, "period_days": days}

    def get_top_search_queries(self, limit: int = 10) -> list:
        """Get most popular text search queries."""
        if not self.is_connected():
            return []

        pipeline = [
            {"$match": {"type": "text"}},
            {"$group": {"_id": {"$toLower": "$query"}, "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        results = list(self.logs.aggregate(pipeline))
        return [{"query": r["_id"], "count": r["count"]} for r in results]

    def get_popular_wall_colors(self, limit: int = 10) -> list:
        """Get most common wall colors from image uploads."""
        if not self.is_connected():
            return []

        pipeline = [
            {"$match": {"type": "image"}},
            {"$unwind": "$wall_colors"},
            {"$group": {"_id": "$wall_colors", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        results = list(self.logs.aggregate(pipeline))
        return [{"color": r["_id"], "count": r["count"]} for r in results]

    def get_most_recommended_paintings(self, limit: int = 10) -> list:
        """Get paintings that appear most in recommendations."""
        if not self.is_connected():
            return []

        pipeline = [
            {"$unwind": "$top_results"},
            {"$group": {
                "_id": "$top_results.product_id",
                "title": {"$first": "$top_results.title"},
                "times_recommended": {"$sum": 1},
                "avg_score": {"$avg": "$top_results.score"},
            }},
            {"$sort": {"times_recommended": -1}},
            {"$limit": limit},
        ]
        results = list(self.logs.aggregate(pipeline))
        return [
            {
                "product_id": r["_id"],
                "title": r["title"],
                "times_recommended": r["times_recommended"],
                "avg_score": round(r["avg_score"], 4),
            }
            for r in results
        ]

    def get_platform_stats(self) -> list:
        """Get usage split by platform (web vs app)."""
        if not self.is_connected():
            return []

        pipeline = [
            {"$group": {"_id": "$platform", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        results = list(self.logs.aggregate(pipeline))
        return [{"platform": r["_id"], "count": r["count"]} for r in results]

    def get_hourly_usage(self) -> list:
        """Get recommendation counts by hour of day."""
        if not self.is_connected():
            return []

        pipeline = [
            {"$group": {
                "_id": {"$hour": "$timestamp"},
                "count": {"$sum": 1},
            }},
            {"$sort": {"_id": 1}},
        ]
        results = list(self.logs.aggregate(pipeline))
        return [{"hour": r["_id"], "count": r["count"]} for r in results]

    def get_daily_usage(self, days: int = 30) -> list:
        """Get recommendation counts per day for the last N days."""
        if not self.is_connected():
            return []

        since = datetime.utcnow() - timedelta(days=days)
        pipeline = [
            {"$match": {"timestamp": {"$gte": since}}},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                "count": {"$sum": 1},
            }},
            {"$sort": {"_id": 1}},
        ]
        results = list(self.logs.aggregate(pipeline))
        return [{"date": r["_id"], "count": r["count"]} for r in results]

    def get_recent_logs(self, limit: int = 20) -> list:
        """Get the most recent recommendation logs."""
        if not self.is_connected():
            return []

        results = list(
            self.logs.find({}, {"_id": 0})
            .sort("timestamp", -1)
            .limit(limit)
        )
        for r in results:
            r["timestamp"] = r["timestamp"].isoformat()
        return results