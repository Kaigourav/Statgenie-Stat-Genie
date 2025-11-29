"""
Job Storage Module
Redis-backed persistent storage with compression, TTL, and in-memory fallback.
"""
import json
import zlib
import base64
import uuid
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️  Redis not installed. Using in-memory fallback only.")

from config import config


class JobStorage:
    """
    Persistent job storage with Redis primary + in-memory fallback.
    Supports compression, TTL, and safe serialization.
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.memory_store: Dict[str, Dict[str, Any]] = {}
        self.using_redis = False
        
        # Initialize Redis connection
        if REDIS_AVAILABLE:
            self._init_redis()
        else:
            print("📦 Using in-memory storage (Redis unavailable)")
    
    def _init_redis(self):
        """Initialize Redis connection with retry logic"""
        try:
            # SSL parameters for Redis Cloud
            ssl_params = {}
            if config.storage.REDIS_SSL:
                import ssl
                ssl_params = {
                    'ssl': True,
                    'ssl_cert_reqs': ssl.CERT_NONE,  # Bypass certificate verification
                    'ssl_check_hostname': False
                }
            
            self.redis_client = redis.Redis(
                host=config.storage.REDIS_HOST,
                port=config.storage.REDIS_PORT,
                db=config.storage.REDIS_DB,
                password=config.storage.REDIS_PASSWORD,
                decode_responses=False,  # We handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                **ssl_params
            )
            # Test connection
            self.redis_client.ping()
            self.using_redis = True
            print(f"✅ Redis connected: {config.storage.REDIS_HOST}:{config.storage.REDIS_PORT}")
        except Exception as e:
            print(f"⚠️  Redis connection failed: {e}")
            if config.storage.FALLBACK_TO_MEMORY:
                print("📦 Falling back to in-memory storage")
                self.using_redis = False
            else:
                raise RuntimeError("Redis unavailable and fallback disabled")
    
    def _compress(self, data: bytes) -> bytes:
        """Compress data if enabled and above threshold"""
        if not config.storage.USE_COMPRESSION:
            return data
        
        if len(data) < config.storage.COMPRESSION_THRESHOLD_BYTES:
            return data
        
        compressed = zlib.compress(data, level=6)
        # Only use compressed if actually smaller
        return compressed if len(compressed) < len(data) else data
    
    def _decompress(self, data: bytes) -> bytes:
        """Attempt decompression, return original if not compressed"""
        try:
            return zlib.decompress(data)
        except zlib.error:
            # Not compressed, return as-is
            return data
    
    def _serialize_dataframe(self, df: pd.DataFrame) -> str:
        """Serialize DataFrame to JSON string"""
        # Convert to records format for compact storage
        return json.dumps(df.to_dict(orient='records'))
    
    def _deserialize_dataframe(self, data: str) -> pd.DataFrame:
        """Deserialize JSON string to DataFrame"""
        records = json.loads(data)
        return pd.DataFrame(records)
    
    def _make_key(self, job_id: str, suffix: str = "") -> str:
        """Generate Redis key with prefix"""
        prefix = "statgenie:job:"
        return f"{prefix}{job_id}" + (f":{suffix}" if suffix else "")
    
    def create_job(self) -> str:
        """Generate new job ID"""
        return str(uuid.uuid4())
    
    def save_job(self, job_id: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Save job data with optional TTL.
        
        Args:
            job_id: Unique job identifier
            data: Job data dictionary (must be JSON-serializable)
            ttl: Time-to-live in seconds (default from config)
        
        Returns:
            True if saved successfully
        """
        ttl = ttl or config.storage.JOB_TTL_SECONDS
        
        # Prepare metadata
        job_data = {
            "job_id": job_id,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(seconds=ttl)).isoformat(),
            "data": data
        }
        
        try:
            if self.using_redis and self.redis_client:
                # Serialize and compress
                serialized = json.dumps(job_data).encode('utf-8')
                compressed = self._compress(serialized)
                
                # Store in Redis with TTL
                key = self._make_key(job_id)
                self.redis_client.setex(key, ttl, compressed)
                return True
            else:
                # In-memory fallback
                self.memory_store[job_id] = job_data
                return True
        
        except Exception as e:
            print(f"❌ Failed to save job {job_id}: {e}")
            # Fallback to memory if Redis fails
            if config.storage.FALLBACK_TO_MEMORY:
                self.memory_store[job_id] = job_data
                return True
            return False
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve job data by ID.
        
        Args:
            job_id: Job identifier
        
        Returns:
            Job data dictionary or None if not found
        """
        try:
            if self.using_redis and self.redis_client:
                key = self._make_key(job_id)
                compressed = self.redis_client.get(key)
                
                if compressed is None:
                    return None
                
                # Decompress and deserialize
                decompressed = self._decompress(compressed)
                job_data = json.loads(decompressed.decode('utf-8'))
                return job_data.get("data")
            
            else:
                # In-memory fallback
                job_data = self.memory_store.get(job_id)
                return job_data.get("data") if job_data else None
        
        except Exception as e:
            print(f"❌ Failed to retrieve job {job_id}: {e}")
            # Try memory fallback
            if job_id in self.memory_store:
                job_data = self.memory_store[job_id]
                return job_data.get("data")
            return None
    
    def delete_job(self, job_id: str) -> bool:
        """Delete job from storage"""
        try:
            if self.using_redis and self.redis_client:
                key = self._make_key(job_id)
                self.redis_client.delete(key)
            
            # Also remove from memory if present
            if job_id in self.memory_store:
                del self.memory_store[job_id]
            
            return True
        
        except Exception as e:
            print(f"❌ Failed to delete job {job_id}: {e}")
            return False
    
    def job_exists(self, job_id: str) -> bool:
        """Check if job exists in storage"""
        try:
            if self.using_redis and self.redis_client:
                key = self._make_key(job_id)
                return bool(self.redis_client.exists(key))
            else:
                return job_id in self.memory_store
        except Exception:
            return job_id in self.memory_store
    
    def extend_ttl(self, job_id: str, additional_seconds: int = 3600) -> bool:
        """Extend job TTL by additional seconds"""
        try:
            if self.using_redis and self.redis_client:
                key = self._make_key(job_id)
                current_ttl = self.redis_client.ttl(key)
                if current_ttl > 0:
                    new_ttl = current_ttl + additional_seconds
                    self.redis_client.expire(key, new_ttl)
                    return True
            return False
        except Exception as e:
            print(f"❌ Failed to extend TTL for job {job_id}: {e}")
            return False
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired jobs from memory store.
        Redis handles expiration automatically.
        
        Returns:
            Number of jobs cleaned up
        """
        if not self.memory_store:
            return 0
        
        now = datetime.utcnow()
        expired_jobs = []
        
        for job_id, job_data in self.memory_store.items():
            expires_at_str = job_data.get("expires_at")
            if expires_at_str:
                try:
                    expires_at = datetime.fromisoformat(expires_at_str)
                    if now > expires_at:
                        expired_jobs.append(job_id)
                except Exception:
                    pass
        
        for job_id in expired_jobs:
            del self.memory_store[job_id]
        
        if expired_jobs:
            print(f"🧹 Cleaned up {len(expired_jobs)} expired jobs from memory")
        
        return len(expired_jobs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            "using_redis": self.using_redis,
            "memory_jobs": len(self.memory_store),
            "redis_connected": False
        }
        
        if self.using_redis and self.redis_client:
            try:
                info = self.redis_client.info()
                stats["redis_connected"] = True
                stats["redis_used_memory_mb"] = round(info.get("used_memory", 0) / 1024 / 1024, 2)
                stats["redis_keys"] = self.redis_client.dbsize()
            except Exception as e:
                stats["redis_error"] = str(e)
        
        return stats


# Global storage instance
job_storage = JobStorage()
