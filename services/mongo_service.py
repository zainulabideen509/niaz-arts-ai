"""
MongoDB Service - Logs recommendations, feedback, and provides analytics.
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
COLLECTION_FEEDBACK = "recommendation_feedback"


class MongoService:
    def __init__(self):
        self.client = None
        self.db = None
        self.logs = None
        self.feedback = None

        if not MONGO_URI:
            logger.warning("MONGO_URI not set! Logging disabled.")
            return

        try:
            self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            self.client.admin.command("ping")
            self.db = self.client[DB_NAME]
            self.logs = self.db[COLLECTION_LOGS]
            self.feedback = self.db[COLLECTION_FEEDBACK]
            logger.info("Connected to MongoDB Atlas successfully.")
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            self.client = None

    def is_connected(self) -> bool:
        return self.client is not None and self.logs is not None

    # ── LOGGING ──────────────────────────────────────────────

    def log_image_recommendation(self, wall_colors, results, platform="unknown"):
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
        except Exception as e:
            logger.error(f"Failed to log: {e}")

    def log_text_recommendation(self, query, results, platform="unknown"):
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
        except Exception as e:
            logger.error(f"Failed to log: {e}")

    # ── FEEDBACK ─────────────────────────────────────────────

    def save_feedback(self, product_id, title, feedback_type, query="", platform="unknown"):
        """Save customer feedback (thumbs up/down) on a recommendation."""
        if not self.is_connected():
            return
        try:
            doc = {
                "product_id": product_id,
                "title": title,
                "feedback": feedback_type,
                "query": query,
                "platform": platform,
                "timestamp": datetime.utcnow(),
            }
            self.feedback.insert_one(doc)
            logger.info(f"Feedback saved: {feedback_type} for {title}")
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")

    def get_feedback_stats(self):
        """Get overall feedback statistics."""
        if not self.is_connected():
            return {"total": 0, "likes": 0, "dislikes": 0}
        try:
            total = self.feedback.count_documents({})
            likes = self.feedback.count_documents({"feedback": "like"})
            dislikes = self.feedback.count_documents({"feedback": "dislike"})
            return {
                "total": total,
                "likes": likes,
                "dislikes": dislikes,
                "satisfaction_rate": round((likes / total * 100), 1) if total > 0 else 0,
            }
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}")
            return {"total": 0, "likes": 0, "dislikes": 0}

    def get_most_liked_paintings(self, limit=10):
        """Get paintings with most thumbs up."""
        if not self.is_connected():
            return []
        pipeline = [
            {"$match": {"feedback": "like"}},
            {"$group": {
                "_id": "$product_id",
                "title": {"$first": "$title"},
                "likes": {"$sum": 1},
            }},
            {"$sort": {"likes": -1}},
            {"$limit": limit},
        ]
        results = list(self.feedback.aggregate(pipeline))
        return [{"product_id": r["_id"], "title": r["title"], "likes": r["likes"]} for r in results]

    def get_most_disliked_paintings(self, limit=10):
        """Get paintings with most thumbs down."""
        if not self.is_connected():
            return []
        pipeline = [
            {"$match": {"feedback": "dislike"}},
            {"$group": {
                "_id": "$product_id",
                "title": {"$first": "$title"},
                "dislikes": {"$sum": 1},
            }},
            {"$sort": {"dislikes": -1}},
            {"$limit": limit},
        ]
        results = list(self.feedback.aggregate(pipeline))
        return [{"product_id": r["_id"], "title": r["title"], "dislikes": r["dislikes"]} for r in results]

    def get_recent_feedback(self, limit=20):
        """Get most recent feedback entries."""
        if not self.is_connected():
            return []
        results = list(
            self.feedback.find({}, {"_id": 0})
            .sort("timestamp", -1)
            .limit(limit)
        )
        for r in results:
            r["timestamp"] = r["timestamp"].isoformat()
        return results

    # ── ANALYTICS ────────────────────────────────────────────

    def get_total_recommendations(self, days=30):
        if not self.is_connected():
            return {"total": 0, "image": 0, "text": 0}
        since = datetime.utcnow() - timedelta(days=days)
        total = self.logs.count_documents({"timestamp": {"$gte": since}})
        image = self.logs.count_documents({"type": "image", "timestamp": {"$gte": since}})
        text = self.logs.count_documents({"type": "text", "timestamp": {"$gte": since}})
        return {"total": total, "image": image, "text": text, "period_days": days}

    def get_top_search_queries(self, limit=10):
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

    def get_popular_wall_colors(self, limit=10):
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

    def get_most_recommended_paintings(self, limit=10):
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

    def get_platform_stats(self):
        if not self.is_connected():
            return []
        pipeline = [
            {"$group": {"_id": "$platform", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        results = list(self.logs.aggregate(pipeline))
        return [{"platform": r["_id"], "count": r["count"]} for r in results]

    def get_hourly_usage(self):
        if not self.is_connected():
            return []
        pipeline = [
            {"$group": {"_id": {"$hour": "$timestamp"}, "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ]
        results = list(self.logs.aggregate(pipeline))
        return [{"hour": r["_id"], "count": r["count"]} for r in results]

    def get_daily_usage(self, days=30):
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

    def get_recent_logs(self, limit=20):
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