"""
Minimal Redis Streams producer.
Run from host:  python tests/redis_producer.py
Requires: pip install redis
"""
import time
import redis

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

stream = "nhl:ingest"
for i in range(5):
    msg_id = r.xadd(stream, {"source": "test", "text": f"message {i}", "ts": str(time.time())})
    print(f"produced {msg_id}")

print("done — run redis_consumer.py to read them back")
