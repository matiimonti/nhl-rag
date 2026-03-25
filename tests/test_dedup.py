"""
Unit tests for services/ingestion/dedup.py

Coverage:
- _hash: determinism, sensitivity to source/payload/timestamp changes, key ordering
- is_duplicate: correct Redis exists() interpretation
- mark_seen: correct key, correct TTL per source
- check_and_mark: atomic SET NX semantics, identical docs skipped,
                  near-duplicates with different timestamps not skipped
"""
import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "ingestion"))

import pytest
from unittest.mock import AsyncMock, MagicMock

from dedup import (
    DEDUP_PREFIX,
    Deduplicator,
    _DEFAULT_TTL,
    _TTL,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def _make_dedup(*, exists_return=0, set_return=1):
    """Return a Deduplicator backed by a fully mocked async Redis client."""
    redis = AsyncMock()
    redis.exists = AsyncMock(return_value=exists_return)
    redis.setex = AsyncMock(return_value=True)
    # SET NX: returns the value on insert, None if key already existed
    redis.set = AsyncMock(return_value=set_return if set_return else None)
    return Deduplicator(redis), redis


# ---------------------------------------------------------------------------
# _hash
# ---------------------------------------------------------------------------

class TestHash:
    def setup_method(self):
        self.dedup, _ = _make_dedup()

    def test_same_input_produces_same_hash(self):
        payload = {"title": "story", "date": "2025-01-01"}
        assert self.dedup._hash("news", payload) == self.dedup._hash("news", payload)

    def test_different_source_produces_different_hash(self):
        payload = {"title": "story"}
        assert self.dedup._hash("news", payload) != self.dedup._hash("reddit_hockey", payload)

    def test_different_payload_produces_different_hash(self):
        assert (
            self.dedup._hash("news", {"title": "A"})
            != self.dedup._hash("news", {"title": "B"})
        )

    def test_key_order_does_not_affect_hash(self):
        """sort_keys=True means payload key order is irrelevant."""
        p1 = {"b": 2, "a": 1}
        p2 = {"a": 1, "b": 2}
        assert self.dedup._hash("news", p1) == self.dedup._hash("news", p2)

    def test_hash_is_64_char_hex_string(self):
        h = self.dedup._hash("news", {"x": 1})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_timestamp_change_changes_hash(self):
        """Near-duplicate: same content, different timestamp → different hash."""
        base = {"title": "Bruins win", "score": "3-2"}
        with_ts_1 = {**base, "published_at": "2025-01-01T10:00:00"}
        with_ts_2 = {**base, "published_at": "2025-01-01T11:00:00"}
        assert self.dedup._hash("news", with_ts_1) != self.dedup._hash("news", with_ts_2)

    def test_hash_prefix_included_in_redis_key(self):
        h = self.dedup._hash("news", {"title": "x"})
        expected_key = DEDUP_PREFIX + h
        assert expected_key.startswith("dedup:")


# ---------------------------------------------------------------------------
# is_duplicate
# ---------------------------------------------------------------------------

class TestIsDuplicate:
    def test_returns_false_when_key_absent(self):
        dedup, _ = _make_dedup(exists_return=0)
        result = _run(dedup.is_duplicate("news", {"title": "new story"}))
        assert result is False

    def test_returns_true_when_key_present(self):
        dedup, _ = _make_dedup(exists_return=1)
        result = _run(dedup.is_duplicate("news", {"title": "seen story"}))
        assert result is True

    def test_calls_exists_with_correct_key(self):
        dedup, redis = _make_dedup(exists_return=0)
        payload = {"title": "story"}
        _run(dedup.is_duplicate("news", payload))
        expected_key = DEDUP_PREFIX + dedup._hash("news", payload)
        redis.exists.assert_called_once_with(expected_key)


# ---------------------------------------------------------------------------
# mark_seen
# ---------------------------------------------------------------------------

class TestMarkSeen:
    def test_calls_setex_with_correct_key(self):
        dedup, redis = _make_dedup()
        payload = {"title": "story"}
        _run(dedup.mark_seen("news", payload))
        expected_key = DEDUP_PREFIX + dedup._hash("news", payload)
        args = redis.setex.call_args
        assert args[0][0] == expected_key

    def test_setex_value_is_one(self):
        dedup, redis = _make_dedup()
        _run(dedup.mark_seen("news", {"title": "x"}))
        args = redis.setex.call_args[0]
        assert args[2] == "1"

    @pytest.mark.parametrize("source,expected_ttl", [
        ("nhl_play_by_play", 6 * 3600),
        ("nhl_scores",       3600),
        ("nhl_standings",    3600),
        ("nhl_player_stats", 3600),
        ("reddit_hockey",    24 * 3600),
        ("reddit_nhl",       24 * 3600),
        ("news",             24 * 3600),
    ])
    def test_correct_ttl_per_source(self, source, expected_ttl):
        dedup, redis = _make_dedup()
        _run(dedup.mark_seen(source, {"x": 1}))
        ttl_used = redis.setex.call_args[0][1]
        assert ttl_used == expected_ttl

    def test_unknown_source_uses_default_ttl(self):
        dedup, redis = _make_dedup()
        _run(dedup.mark_seen("unknown_source", {"x": 1}))
        ttl_used = redis.setex.call_args[0][1]
        assert ttl_used == _DEFAULT_TTL


# ---------------------------------------------------------------------------
# check_and_mark — atomic SET NX
# ---------------------------------------------------------------------------

class TestCheckAndMark:
    def test_new_document_not_duplicate(self):
        """SET NX succeeds (returns truthy) → document is new."""
        dedup, _ = _make_dedup(set_return=1)
        result = _run(dedup.check_and_mark("news", {"title": "new"}))
        assert result is False

    def test_existing_document_is_duplicate(self):
        """SET NX returns None (key already existed) → document is duplicate."""
        dedup, _ = _make_dedup(set_return=None)
        result = _run(dedup.check_and_mark("news", {"title": "seen"}))
        assert result is True

    def test_set_called_with_nx_flag(self):
        dedup, redis = _make_dedup(set_return=1)
        payload = {"title": "story"}
        _run(dedup.check_and_mark("news", payload))
        _, kwargs = redis.set.call_args
        assert kwargs.get("nx") is True

    def test_set_called_with_correct_ttl(self):
        dedup, redis = _make_dedup(set_return=1)
        _run(dedup.check_and_mark("reddit_hockey", {"title": "x"}))
        _, kwargs = redis.set.call_args
        assert kwargs.get("ex") == 24 * 3600

    def test_set_called_with_correct_key(self):
        dedup, redis = _make_dedup(set_return=1)
        payload = {"title": "story"}
        _run(dedup.check_and_mark("news", payload))
        expected_key = DEDUP_PREFIX + dedup._hash("news", payload)
        args, _ = redis.set.call_args
        assert args[0] == expected_key

    # --- Core requirements from the task spec ---

    def test_identical_documents_are_skipped(self):
        """
        Simulate two arrivals of the exact same document.
        First call: SET NX succeeds → not a dup.
        Second call: SET NX returns None → is a dup.
        """
        redis = AsyncMock()
        # First call inserts, second call finds key already present
        redis.set = AsyncMock(side_effect=[1, None])
        dedup = Deduplicator(redis)
        payload = {"title": "Bruins win 3-2", "game_id": "2025020001"}

        first = _run(dedup.check_and_mark("nhl_scores", payload))
        second = _run(dedup.check_and_mark("nhl_scores", payload))

        assert first is False   # new
        assert second is True   # duplicate — skipped

    def test_near_duplicate_different_timestamp_not_skipped(self):
        """
        Two documents with identical content but different timestamps
        must hash differently and both be treated as new.
        """
        redis = AsyncMock()
        redis.set = AsyncMock(return_value=1)  # always inserts (keys differ)
        dedup = Deduplicator(redis)

        base = {"title": "Bruins win", "score": "3-2"}
        payload_v1 = {**base, "published_at": "2025-01-01T10:00:00"}
        payload_v2 = {**base, "published_at": "2025-01-01T11:00:00"}

        result_v1 = _run(dedup.check_and_mark("news", payload_v1))
        result_v2 = _run(dedup.check_and_mark("news", payload_v2))

        assert result_v1 is False  # new
        assert result_v2 is False  # also new — different timestamp = different doc

        # Confirm the two calls used different Redis keys
        call_keys = [call_args[0][0] for call_args in redis.set.call_args_list]
        assert call_keys[0] != call_keys[1]

    def test_same_payload_different_source_not_duplicate(self):
        """Same payload text under a different source is a distinct document."""
        redis = AsyncMock()
        redis.set = AsyncMock(return_value=1)
        dedup = Deduplicator(redis)
        payload = {"title": "Game recap"}

        r1 = _run(dedup.check_and_mark("reddit_hockey", payload))
        r2 = _run(dedup.check_and_mark("reddit_nhl", payload))

        assert r1 is False
        assert r2 is False
        call_keys = [c[0][0] for c in redis.set.call_args_list]
        assert call_keys[0] != call_keys[1]
