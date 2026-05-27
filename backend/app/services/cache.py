import json
import logging
from typing import Optional, Dict, Any
import redis.asyncio as aioredis
from app.config import settings

logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.redis_url = settings.REDIS_URL
        
    async def connect(self):
        """Initialize Redis Connection Pool"""
        try:
            self.redis = aioredis.from_url(
                self.redis_url, 
                encoding="utf-8", 
                decode_responses=True
            )
            logger.info("Successfully connected to Redis cache.")
        except Exception as e:
            logger.error(f"Failed to connect to Redis cache: {e}")
            self.redis = None
            
    async def close(self):
        """Close connection"""
        if self.redis:
            await self.redis.close()
            
    def _normalize_key(self, query: str) -> str:
        """Helper to create a clean, lowercase cache key"""
        return f"query_cache:{query.strip().lower()}"
        
    async def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached query response if available"""
        if not self.redis:
            return None
        try:
            key = self._normalize_key(query)
            cached_val = await self.redis.get(key)
            if cached_val:
                logger.info(f"Cache Hit for query: {query}")
                return json.loads(cached_val)
        except Exception as e:
            logger.warning(f"Failed to fetch cache for query '{query}': {e}")
        return None

    async def set(self, query: str, response_data: Dict[str, Any], expire_seconds: int = 3600):
        """Cache a query response with an optional expiry"""
        if not self.redis:
            return
        try:
            key = self._normalize_key(query)
            await self.redis.setex(
                key,
                expire_seconds,
                json.dumps(response_data)
            )
            logger.info(f"Cached response for query: {query}")
        except Exception as e:
            logger.warning(f"Failed to cache response for query '{query}': {e}")

# Single global instance
cache_service = RedisCache()
