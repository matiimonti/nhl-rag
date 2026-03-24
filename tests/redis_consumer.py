"""
Minimal Redis Streams consumer.
Run from host:  python tests/redis_consumer.py
Requires: pip install redis
"""
import redis

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

stream = "nhl:ingest"
messages = r.xrange(stream)

if not messages:
    print("no messages found — run redis_producer.py first")
else:
    for msg_id, fields in messages:
        print(f"{msg_id}: {fields}")
    print(f"\n{len(messages)} message(s) read from '{stream}'")
