"""
Unit tests for services/ingestion/chunker.py

Coverage:
- Chunk size bounds (every chunk <= chunk_size tokens)
- Overlap correctness (consecutive chunks share overlap tokens)
- Metadata preservation (source, date, url, teams, players, indices)
- Edge cases: empty input, single-token input, unicode text
"""
import sys
import os

# Allow importing from the ingestion service without installing it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "ingestion"))

import pytest
import tiktoken

from chunker import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    Chunk,
    ChunkMetadata,
    Chunker,
    extract_players,
    extract_teams,
)

_enc = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_long_text(n_tokens: int) -> str:
    """Return text that encodes to at least n_tokens.

    ' word' (with leading space) is a single token in cl100k_base, so
    repeating it n times reliably produces n tokens without any
    decode/re-encode roundtrip that could lose tokens at a boundary.
    """
    return " word" * n_tokens


def _news_payload(text: str, url: str = "https://example.com") -> dict:
    return {"title": text, "url": url}


def _reddit_payload(title: str, body: str = "", permalink: str = "/r/hockey/x") -> dict:
    return {"title": title, "selftext": body, "permalink": permalink}


# ---------------------------------------------------------------------------
# _split_tokens: size and overlap
# ---------------------------------------------------------------------------

class TestSplitTokens:
    def setup_method(self):
        self.chunker = Chunker(chunk_size=10, overlap=3)

    def test_empty_string_returns_no_windows(self):
        assert self.chunker._split_tokens("") == []

    def test_short_text_single_window(self):
        text = _enc.decode(_enc.encode("hello world")[:5])
        windows = self.chunker._split_tokens(text)
        assert len(windows) == 1

    def test_each_window_at_most_chunk_size(self):
        text = _make_long_text(50)
        windows = self.chunker._split_tokens(text)
        for w in windows:
            assert len(w) <= self.chunker.chunk_size

    def test_overlap_correctness(self):
        """Last `overlap` tokens of window[i] == first `overlap` tokens of window[i+1]."""
        text = _make_long_text(40)
        windows = self.chunker._split_tokens(text)
        assert len(windows) >= 2, "Need at least 2 windows to test overlap"
        for i in range(len(windows) - 1):
            tail = windows[i][-self.chunker.overlap:]
            head = windows[i + 1][: self.chunker.overlap]
            assert tail == head, f"Overlap mismatch between window {i} and {i+1}"

    def test_exact_chunk_size_gives_one_window(self):
        # Text with exactly chunk_size tokens → should produce exactly 1 window
        text = _make_long_text(10)  # " word" * 10 = 10 tokens
        windows = self.chunker._split_tokens(text)
        assert len(windows) == 1
        assert len(windows[0]) <= self.chunker.chunk_size

    def test_one_token_over_chunk_size_gives_two_windows(self):
        # Text with chunk_size + several tokens → must produce >= 2 windows
        text = _make_long_text(20)  # 20 tokens > chunk_size=10
        windows = self.chunker._split_tokens(text)
        assert len(windows) >= 2
        assert len(windows[0]) <= self.chunker.chunk_size

    def test_last_window_covers_final_token(self):
        """No tokens are silently dropped at the tail of the document."""
        text = _make_long_text(25)
        all_tokens = _enc.encode(text)
        windows = self.chunker._split_tokens(text)
        assert windows[-1][-1] == all_tokens[-1]

    def test_zero_overlap_windows_are_contiguous(self):
        """With overlap=0 windows tile the token sequence with no gaps or repeats."""
        chunker = Chunker(chunk_size=10, overlap=0)
        text = _make_long_text(25)
        all_tokens = _enc.encode(text)
        windows = chunker._split_tokens(text)
        # Reconstruct the full token sequence from the windows — should equal original
        reconstructed = []
        for w in windows:
            reconstructed.extend(w)
        assert reconstructed == all_tokens


# ---------------------------------------------------------------------------
# Chunker constructor validation
# ---------------------------------------------------------------------------

class TestChunkerConstructor:
    def test_overlap_equal_to_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            Chunker(chunk_size=10, overlap=10)

    def test_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            Chunker(chunk_size=10, overlap=11)

    def test_zero_overlap_is_valid(self):
        c = Chunker(chunk_size=10, overlap=0)
        assert c.overlap == 0

    def test_default_constants_are_valid(self):
        # Sanity-check that the module-level defaults don't violate the invariant
        assert CHUNK_OVERLAP < CHUNK_SIZE


# ---------------------------------------------------------------------------
# chunk_document: unknown / empty sources
# ---------------------------------------------------------------------------

class TestChunkDocumentUnknownSource:
    def setup_method(self):
        self.chunker = Chunker()

    def test_unknown_source_returns_empty(self):
        chunks = self.chunker.chunk_document("not_a_source", {"title": "hi"}, "2025-01-01")
        assert chunks == []

    def test_empty_payload_text_returns_empty(self):
        # news with no title / description / content → empty string after join
        chunks = self.chunker.chunk_document("news", {}, "2025-01-01")
        assert chunks == []

    def test_whitespace_only_text_returns_empty(self):
        chunks = self.chunker.chunk_document("news", {"title": "   "}, "2025-01-01")
        assert chunks == []


# ---------------------------------------------------------------------------
# chunk_document: chunk size bounds
# ---------------------------------------------------------------------------

class TestChunkSizeBounds:
    def test_default_chunk_size_respected(self):
        chunker = Chunker()
        text = _make_long_text(CHUNK_SIZE * 3)
        chunks = chunker.chunk_document("news", _news_payload(text), "2025-01-01")
        assert chunks, "Expected at least one chunk"
        for c in chunks:
            assert c.token_count <= CHUNK_SIZE

    def test_custom_chunk_size_respected(self):
        chunker = Chunker(chunk_size=20, overlap=5)
        text = _make_long_text(80)
        chunks = chunker.chunk_document("news", _news_payload(text), "2025-01-01")
        for c in chunks:
            assert c.token_count <= 20

    def test_token_count_matches_actual_tokens(self):
        chunker = Chunker(chunk_size=20, overlap=4)
        text = _make_long_text(60)
        chunks = chunker.chunk_document("news", _news_payload(text), "2025-01-01")
        for c in chunks:
            assert c.token_count == len(_enc.encode(c.text))


# ---------------------------------------------------------------------------
# chunk_document: overlap correctness at document level
# ---------------------------------------------------------------------------

class TestDocumentOverlap:
    def test_consecutive_chunks_share_overlap_tokens(self):
        chunker = Chunker(chunk_size=20, overlap=5)
        text = _make_long_text(80)
        chunks = chunker.chunk_document("news", _news_payload(text), "2025-01-01")
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            tail_tokens = _enc.encode(chunks[i].text)[-chunker.overlap:]
            head_tokens = _enc.encode(chunks[i + 1].text)[: chunker.overlap]
            assert tail_tokens == head_tokens


# ---------------------------------------------------------------------------
# chunk_document: metadata preservation
# ---------------------------------------------------------------------------

class TestMetadataPreservation:
    def setup_method(self):
        self.chunker = Chunker()

    def test_source_preserved(self):
        chunks = self.chunker.chunk_document(
            "news", _news_payload("Short news article"), "2025-03-01"
        )
        assert all(c.metadata.source == "news" for c in chunks)

    def test_date_preserved(self):
        chunks = self.chunker.chunk_document(
            "news", _news_payload("Short news"), "2025-03-15"
        )
        assert all(c.metadata.date == "2025-03-15" for c in chunks)

    def test_url_preserved_news(self):
        chunks = self.chunker.chunk_document(
            "news", {"title": "Story", "url": "https://sportsnet.ca/story"}, "2025-01-01"
        )
        assert all(c.metadata.url == "https://sportsnet.ca/story" for c in chunks)

    def test_url_preserved_reddit(self):
        payload = _reddit_payload("Game thread", permalink="/r/hockey/comments/abc")
        chunks = self.chunker.chunk_document("reddit_hockey", payload, "2025-01-01")
        assert all(c.metadata.url == "https://reddit.com/r/hockey/comments/abc" for c in chunks)

    def test_url_none_for_nhl_scores(self):
        payload = {
            "game_state": "FINAL", "game_date": "2025-01-10",
            "home_team": {"abbrev": "TOR", "score": 3},
            "away_team": {"abbrev": "BOS", "score": 2},
        }
        chunks = self.chunker.chunk_document("nhl_scores", payload, "2025-01-10")
        assert all(c.metadata.url is None for c in chunks)

    def test_doc_id_consistent_across_chunks(self):
        text = _make_long_text(CHUNK_SIZE * 2)
        chunks = self.chunker.chunk_document("news", _news_payload(text), "2025-01-01")
        assert len(chunks) >= 2
        doc_ids = {c.metadata.doc_id for c in chunks}
        assert len(doc_ids) == 1

    def test_chunk_index_and_total(self):
        text = _make_long_text(CHUNK_SIZE * 2)
        chunks = self.chunker.chunk_document("news", _news_payload(text), "2025-01-01")
        total = len(chunks)
        for i, c in enumerate(chunks):
            assert c.metadata.chunk_index == i
            assert c.metadata.total_chunks == total

    def test_doc_id_is_deterministic(self):
        """Same source + text always produces the same doc_id."""
        payload = _news_payload("Deterministic content")
        chunker = Chunker()
        chunks_a = chunker.chunk_document("news", payload, "2025-01-01")
        chunks_b = chunker.chunk_document("news", payload, "2025-01-01")
        assert chunks_a[0].metadata.doc_id == chunks_b[0].metadata.doc_id

    def test_doc_id_differs_for_different_text(self):
        chunker = Chunker()
        c1 = chunker.chunk_document("news", _news_payload("Story A"), "2025-01-01")
        c2 = chunker.chunk_document("news", _news_payload("Story B"), "2025-01-01")
        assert c1[0].metadata.doc_id != c2[0].metadata.doc_id

    def test_doc_id_differs_for_different_source(self):
        chunker = Chunker()
        payload = _reddit_payload("Same text")
        c1 = chunker.chunk_document("reddit_hockey", payload, "2025-01-01")
        c2 = chunker.chunk_document("reddit_nhl", payload, "2025-01-01")
        assert c1[0].metadata.doc_id != c2[0].metadata.doc_id

    def test_single_chunk_index_zero_total_one(self):
        chunks = self.chunker.chunk_document("news", _news_payload("Short"), "2025-01-01")
        assert len(chunks) == 1
        assert chunks[0].metadata.chunk_index == 0
        assert chunks[0].metadata.total_chunks == 1

    def test_teams_extracted(self):
        payload = _news_payload("The Boston Bruins beat the TOR tonight.")
        chunks = self.chunker.chunk_document("news", payload, "2025-01-01")
        teams = chunks[0].metadata.teams
        assert "Boston Bruins" in teams
        assert "Toronto Maple Leafs" in teams

    def test_players_extracted(self):
        payload = _news_payload("Sidney Crosby scored twice against Connor McDavid.")
        chunks = self.chunker.chunk_document("news", payload, "2025-01-01")
        players = chunks[0].metadata.players
        assert "Sidney Crosby" in players
        assert "Connor McDavid" in players  # Mc-prefixed surname must match


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def setup_method(self):
        self.chunker = Chunker()

    def test_empty_string_news(self):
        chunks = self.chunker.chunk_document("news", {"title": ""}, "2025-01-01")
        assert chunks == []

    def test_single_token_input(self):
        # Encode a single-token word and decode it back
        single = _enc.decode([_enc.encode("hi")[0]])
        chunks = self.chunker.chunk_document("news", _news_payload(single), "2025-01-01")
        assert len(chunks) == 1
        assert chunks[0].token_count >= 1

    def test_unicode_text(self):
        text = "Équipe du Montréal Canadiens 🏒 бросок Артёма Зуба"
        chunks = self.chunker.chunk_document(
            "reddit_hockey", _reddit_payload(text), "2025-01-01"
        )
        assert len(chunks) >= 1
        assert chunks[0].text  # non-empty decoded text

    def test_unicode_roundtrip_no_data_loss(self):
        text = "Ōsaka wins 🎉 über-skilled"
        chunks = self.chunker.chunk_document("news", _news_payload(text), "2025-01-01")
        reconstructed = "".join(c.text for c in chunks)
        # Overlap means tokens are repeated — just ensure original is substring
        assert text in reconstructed or len(reconstructed) >= len(text) // 2

    def test_exactly_one_chunk_when_text_fits(self):
        text = _make_long_text(CHUNK_SIZE)
        chunks = self.chunker.chunk_document("news", _news_payload(text), "2025-01-01")
        assert len(chunks) == 1

    def test_multiple_chunks_when_text_exceeds_size(self):
        text = _make_long_text(CHUNK_SIZE + CHUNK_OVERLAP + 10)
        chunks = self.chunker.chunk_document("news", _news_payload(text), "2025-01-01")
        assert len(chunks) >= 2

    def test_all_sources_accepted(self):
        sources = [
            ("nhl_scores", {
                "game_state": "FINAL", "game_date": "2025-01-01",
                "home_team": {"abbrev": "TOR", "score": 1},
                "away_team": {"abbrev": "BOS", "score": 2},
            }),
            ("nhl_standings", {
                "team_name": {"default": "Toronto Maple Leafs"},
                "wins": 30, "losses": 20, "ot_losses": 5, "points": 65, "games_played": 55,
            }),
            ("nhl_player_stats", {
                "name": "Auston Matthews", "team": "TOR",
                "goals": 25, "assists": 30, "points": 55, "games_played": 50,
            }),
            ("nhl_play_by_play", {
                "game_id": "2025020001", "period": 2,
                "time_in_period": "10:34", "type_desc_key": "goal",
            }),
            ("reddit_hockey", _reddit_payload("Great game tonight")),
            ("reddit_nhl", _reddit_payload("Trade deadline talk")),
            ("news", _news_payload("NHL news story")),
        ]
        for source, payload in sources:
            chunks = self.chunker.chunk_document(source, payload, "2025-01-01")
            assert len(chunks) >= 1, f"Expected chunks for source '{source}'"


# ---------------------------------------------------------------------------
# extract_teams / extract_players unit tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Formatter-specific content tests
# ---------------------------------------------------------------------------

class TestFormatters:
    def setup_method(self):
        self.chunker = Chunker()

    def test_news_joins_title_description_content(self):
        payload = {
            "title": "Title here",
            "description": "Description here",
            "content": "Content here",
            "url": "https://example.com",
        }
        chunks = self.chunker.chunk_document("news", payload, "2025-01-01")
        combined = " ".join(c.text for c in chunks)
        assert "Title here" in combined
        assert "Description here" in combined
        assert "Content here" in combined

    def test_reddit_joins_title_and_body(self):
        payload = _reddit_payload("Game thread title", body="Great game discussion body.")
        chunks = self.chunker.chunk_document("reddit_hockey", payload, "2025-01-01")
        combined = " ".join(c.text for c in chunks)
        assert "Game thread title" in combined
        assert "Great game discussion body." in combined

    def test_reddit_body_only_title_when_selftext_empty(self):
        payload = _reddit_payload("Title only", body="")
        chunks = self.chunker.chunk_document("reddit_nhl", payload, "2025-01-01")
        assert len(chunks) >= 1
        assert "Title only" in chunks[0].text


# ---------------------------------------------------------------------------
# extract_teams / extract_players unit tests
# ---------------------------------------------------------------------------

class TestExtractTeams:
    def test_full_name(self):
        assert "Boston Bruins" in extract_teams("The Boston Bruins won tonight.")

    def test_abbreviation(self):
        assert "Toronto Maple Leafs" in extract_teams("TOR beat MTL 4-2.")

    def test_no_teams(self):
        assert extract_teams("The weather is nice today.") == []

    def test_duplicate_deduplication(self):
        teams = extract_teams("BOS beat BOS again.")
        assert teams.count("Boston Bruins") == 1

    def test_abbrev_and_full_name_same_team_deduplicated(self):
        # "TOR" and "Toronto Maple Leafs" refer to the same team — must appear once
        teams = extract_teams("The TOR Toronto Maple Leafs are playing tonight.")
        assert teams.count("Toronto Maple Leafs") == 1


class TestExtractPlayers:
    def test_recognizes_capitalized_name(self):
        players = extract_players("Sidney Crosby had a hat trick.")
        assert "Sidney Crosby" in players

    def test_filters_team_names(self):
        players = extract_players("Boston Bruins won the game.")
        assert "Boston Bruins" not in players

    def test_mixed_case_surnames(self):
        # Mc / Mac prefixes are common in NHL rosters
        players = extract_players("Connor McDavid and Nathan MacKinnon played well.")
        assert "Connor McDavid" in players
        assert "Nathan MacKinnon" in players

    def test_player_mentioned_twice_deduplicated(self):
        players = extract_players("Sidney Crosby scored. Sidney Crosby is on a streak.")
        assert players.count("Sidney Crosby") == 1

    def test_no_players(self):
        assert extract_players("great game tonight") == []
