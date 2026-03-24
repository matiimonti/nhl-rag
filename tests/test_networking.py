"""
Container reachability checks.
Run from host:  python tests/test_networking.py
Requires: pip install redis qdrant-client
"""
import sys
import redis
from qdrant_client import QdrantClient


def check_redis():
    try:
        r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=3)
        r.ping()
        print("Redis:  OK")
        return True
    except Exception as e:
        print(f"Redis:  FAIL — {e}")
        return False


def check_qdrant():
    try:
        client = QdrantClient(host="localhost", port=6333, timeout=3)
        client.get_collections()
        print("Qdrant: OK")
        return True
    except Exception as e:
        print(f"Qdrant: FAIL — {e}")
        return False


if __name__ == "__main__":
    results = [check_redis(), check_qdrant()]
    sys.exit(0 if all(results) else 1)
