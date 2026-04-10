#!/usr/bin/env bash
# Verifies every container can reach its dependencies by service name.
# Run after `docker compose up -d`. Fails loudly on any connection failure.

set -euo pipefail

PASS=0
FAIL=0

check() {
    local from=$1
    local host=$2
    local port=$3

    if docker compose exec "$from" bash -c "timeout 3 bash -c '</dev/tcp/$host/$port'" 2>/dev/null; then
        echo "  PASS  $from → $host:$port"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $from → $host:$port"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Docker networking test ==="

echo ""
echo "ingestion dependencies:"
check ingestion qdrant 6333
check ingestion redis  6379

echo ""
echo "fastapi dependencies:"
check fastapi qdrant 6333
check fastapi redis  6379

echo ""
echo "streamlit dependencies:"
check streamlit fastapi 8000

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
