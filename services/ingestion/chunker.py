"""
Chunking strategy for the NHL RAG ingestion pipeline.

- Fixed-size chunks: 512 tokens
- Overlap: 64 tokens
- Tokeniser: cl100k_base (same encoding as text-embedding-3-*)
- Metadata per chunk: source, date, entity tags (teams, players), url, doc_id
"""
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Optional

import tiktoken

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# cl100k_base is used by OpenAI embedding models and GPT-4
_enc = tiktoken.get_encoding("cl100k_base")

# NHL entity tables

NHL_TEAM_ABBREVS: dict[str, str] = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres",
    "CGY": "Calgary Flames", "CAR": "Carolina Hurricanes", "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche", "CBJ": "Columbus Blue Jackets", "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers", "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild", "MTL": "Montreal Canadiens",
    "NSH": "Nashville Predators", "NJD": "New Jersey Devils", "NYI": "New York Islanders",
    "NYR": "New York Rangers", "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins", "SEA": "Seattle Kraken", "SJS": "San Jose Sharks",
    "STL": "St. Louis Blues", "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Hockey Club", "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights",
    "WSH": "Washington Capitals", "WPG": "Winnipeg Jets",
}

_TEAM_FULL_NAMES = set(NHL_TEAM_ABBREVS.values())
_TEAM_ABBREV_RE = re.compile(r'\b(' + '|'.join(NHL_TEAM_ABBREVS.keys()) + r')\b')
_PLAYER_NAME_RE = re.compile(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b')


# Data classes

@dataclass
class ChunkMetadata:
    source: str
    doc_id: str
    date: str
    url: Optional[str] = None
    teams: list[str] = field(default_factory=list)
    players: list[str] = field(default_factory=list)
    chunk_index: int = 0
    total_chunks: int = 1


@dataclass
class Chunk:
    text: str
    token_count: int
    metadata: ChunkMetadata


# Entity extraction

def extract_teams(text: str) -> list[str]:
    abbrevs = _TEAM_ABBREV_RE.findall(text)
    full = [name for name in _TEAM_FULL_NAMES if name in text]
    return list({NHL_TEAM_ABBREVS.get(a, a) for a in abbrevs} | set(full))


def extract_players(text: str) -> list[str]:
    candidates = _PLAYER_NAME_RE.findall(text)
    # filter out team full names accidentally matched
    return [p for p in candidates if p not in _TEAM_FULL_NAMES]


# Document -> text formatters

def _fmt_nhl_scores(p: dict) -> tuple[str, Optional[str]]:
    home = p.get("home_team", {})
    away = p.get("away_team", {})
    text = (
        f"NHL Game ({p.get('game_state', 'SCHEDULED')}) on {p.get('game_date', '')}. "
        f"{away.get('abbrev', '?')} vs {home.get('abbrev', '?')}. "
        f"Score: {away.get('score', '-')} - {home.get('score', '-')}."
    )
    return text, None


def _fmt_nhl_standings(p: dict) -> tuple[str, Optional[str]]:
    name = p.get("team_name", {}).get("default", "Unknown")
    text = (
        f"{name} standings: {p.get('wins', 0)}W-{p.get('losses', 0)}L-"
        f"{p.get('ot_losses', 0)}OTL, {p.get('points', 0)} points, "
        f"{p.get('games_played', 0)} games played."
    )
    return text, None


def _fmt_nhl_player_stats(p: dict) -> tuple[str, Optional[str]]:
    text = (
        f"{p.get('name', 'Unknown')} ({p.get('team', '?')}): "
        f"{p.get('goals', 0)} goals, {p.get('assists', 0)} assists, "
        f"{p.get('points', 0)} points in {p.get('games_played', 0)} games "
        f"(2024-25 season)."
    )
    return text, None


def _fmt_nhl_play_by_play(p: dict) -> tuple[str, Optional[str]]:
    text = (
        f"Play-by-play event in game {p.get('game_id', '?')}: "
        f"Period {p.get('period', '?')}, {p.get('time_in_period', '?')} — "
        f"{p.get('type_desc_key', 'unknown event')}."
    )
    return text, None


def _fmt_reddit(p: dict) -> tuple[str, Optional[str]]:
    title = p.get("title", "")
    body = p.get("selftext", "").strip()
    text = f"{title}\n\n{body}" if body else title
    url = f"https://reddit.com{p.get('permalink', '')}"
    return text, url


def _fmt_news(p: dict) -> tuple[str, Optional[str]]:
    parts = [p.get("title", "")]
    if p.get("description"):
        parts.append(p["description"])
    if p.get("content"):
        parts.append(p["content"])
    return "\n\n".join(filter(None, parts)), p.get("url")


_FORMATTERS = {
    "nhl_scores": _fmt_nhl_scores,
    "nhl_standings": _fmt_nhl_standings,
    "nhl_player_stats": _fmt_nhl_player_stats,
    "nhl_play_by_play": _fmt_nhl_play_by_play,
    "reddit_hockey": _fmt_reddit,
    "reddit_nhl": _fmt_reddit,
    "news": _fmt_news,
}


# Chunker

class Chunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_tokens(self, text: str) -> list[list[int]]:
        """Split token list into overlapping windows."""
        tokens = _enc.encode(text)
        if not tokens:
            return []
        windows = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            windows.append(tokens[start:end])
            if end >= len(tokens):
                break
            start += self.chunk_size - self.overlap
        return windows

    def chunk_document(self, source: str, payload: dict, date: str) -> list[Chunk]:
        """
        Convert a raw stream document into one or more Chunks.

        Args:
            source: stream source field (e.g. "nhl_scores", "reddit_hockey")
            payload: deserialized document dict
            date: ISO date string of ingestion

        Returns:
            List of Chunk objects ready for embedding.
        """
        formatter = _FORMATTERS.get(source)
        if formatter is None:
            return []

        text, url = formatter(payload)
        if not text.strip():
            return []

        doc_id = hashlib.sha1((source + text).encode()).hexdigest()[:16]
        teams = extract_teams(text)
        players = extract_players(text)

        windows = self._split_tokens(text)
        total = len(windows)

        chunks = []
        for i, token_ids in enumerate(windows):
            chunk_text = _enc.decode(token_ids)
            chunks.append(Chunk(
                text=chunk_text,
                token_count=len(token_ids),
                metadata=ChunkMetadata(
                    source=source,
                    doc_id=doc_id,
                    date=date,
                    url=url,
                    teams=teams,
                    players=players,
                    chunk_index=i,
                    total_chunks=total,
                ),
            ))
        return chunks
