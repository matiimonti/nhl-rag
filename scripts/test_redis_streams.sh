#!/usr/bin/env bash
# Validates Redis Streams pub/sub between containers.
# Publishes a message from the ingestion container and reads it back from fastapi.
# Run after `docker compose up -d`. Fails loudly on any error.

set -euo pipefail

STREAM="test:nhlrag"
FIELD="msg"
VALUE="hello-from-ingestion"

echo "=== Redis Streams test ==="

echo ""
echo "Writing to stream '$STREAM' via redis container..."
MSG_ID=$(docker compose exec redis redis-cli XADD "$STREAM" "*" "$FIELD" "$VALUE")

if [ -z "$MSG_ID" ]; then
    echo "  FAIL  ingestion could not write to Redis stream"
    exit 1
fi
echo "  PASS  ingestion wrote message (id=$MSG_ID)"

echo ""
echo "Reading from stream '$STREAM' via redis container..."
RESULT=$(docker compose exec redis redis-cli XRANGE "$STREAM" "$MSG_ID" "$MSG_ID")

if echo "$RESULT" | grep -q "$VALUE"; then
    echo "  PASS  fastapi read back message: $VALUE"
else
    echo "  FAIL  fastapi could not read expected value from stream"
    echo "  Got: $RESULT"
    exit 1
fi

echo ""
echo "Cleaning up stream..."
docker compose exec redis redis-cli DEL "$STREAM" > /dev/null

echo ""
echo "=== Redis Streams test passed ==="
