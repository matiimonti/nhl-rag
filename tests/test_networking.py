"""
Host→container reachability checks.
Run:  pytest tests/test_networking.py -v
Requires: pip install redis qdrant-client
"""
import redis
import pytest
from qdrant_client import QdrantClient


def test_redis_reachable():
    r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=3)
    assert r.ping(), "Redis did not respond to PING"


def test_qdrant_reachable():
    client = QdrantClient(host="localhost", port=6333, timeout=3)
    # get_collections() raises if unreachable
    client.get_collections()
