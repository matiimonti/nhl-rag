"""
Async NHL API client with typed Pydantic models.

Sources:
  - https://api-web.nhle.com/v1   (scores, standings, play-by-play)
  - https://api.nhle.com/stats/rest/en  (player stats)
"""
import logging
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

WEB_BASE = "https://api-web.nhle.com/v1"
STATS_BASE = "https://api.nhle.com/stats/rest/en"


# Pydantic models

class TeamScore(BaseModel):
    abbrev: str
    score: Optional[int] = None


class Game(BaseModel):
    id: int
    game_date: str = Field(alias="gameDate")
    game_state: str = Field(alias="gameState")
    home_team: TeamScore = Field(alias="homeTeam")
    away_team: TeamScore = Field(alias="awayTeam")

    model_config = {"populate_by_name": True}


class ScoresResponse(BaseModel):
    games: list[Game]


class TeamStanding(BaseModel):
    team_abbrev: dict = Field(alias="teamAbbrev")
    team_name: dict = Field(alias="teamName")
    wins: int
    losses: int
    ot_losses: int = Field(alias="otLosses")
    points: int
    games_played: int = Field(alias="gamesPlayed")

    model_config = {"populate_by_name": True}


class StandingsResponse(BaseModel):
    standings: list[TeamStanding]


class PlayerStat(BaseModel):
    player_id: int = Field(alias="playerId")
    name: str = Field(alias="skaterFullName")
    team: str = Field(alias="teamAbbrevs")
    games_played: int = Field(alias="gamesPlayed")
    goals: int
    assists: int
    points: int

    model_config = {"populate_by_name": True}


class PlayerStatsResponse(BaseModel):
    data: list[PlayerStat]
    total: int


class PlayByPlayEvent(BaseModel):
    event_id: int = Field(alias="eventId")
    period: int
    time_in_period: str = Field(alias="timeInPeriod")
    type_desc_key: str = Field(alias="typeDescKey")
    details: Optional[dict[str, Any]] = None

    model_config = {"populate_by_name": True}


class PlayByPlayResponse(BaseModel):
    id: int
    plays: list[PlayByPlayEvent]


# Client

class NHLClient:
    def __init__(self, timeout: float = 10.0):
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    async def close(self):
        await self._client.aclose()

    async def _get(self, url: str, params: dict | None = None) -> dict:
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    async def fetch_scores(self) -> ScoresResponse:
        """Today's game scores."""
        data = await self._get(f"{WEB_BASE}/score/now")
        return ScoresResponse.model_validate(data)

    async def fetch_standings(self) -> StandingsResponse:
        """Current league standings."""
        data = await self._get(f"{WEB_BASE}/standings/now")
        return StandingsResponse.model_validate(data)

    async def fetch_player_stats(
        self,
        season: str = "20242025",
        limit: int = 100,
    ) -> PlayerStatsResponse:
        """Top skater stats for a given season, sorted by points."""
        data = await self._get(
            f"{STATS_BASE}/skater/summary",
            params={
                "limit": limit,
                "start": 0,
                "sort": "points",
                "cayenneExp": f"seasonId={season}",
            },
        )
        return PlayerStatsResponse.model_validate(data)

    async def fetch_play_by_play(self, game_id: int) -> PlayByPlayResponse:
        """Play-by-play events for a specific game."""
        data = await self._get(f"{WEB_BASE}/gamecenter/{game_id}/play-by-play")
        return PlayByPlayResponse.model_validate(data)

    async def fetch_live_game_ids(self) -> list[int]:
        """Return IDs of games currently in progress (for play-by-play polling)."""
        scores = await self.fetch_scores()
        return [g.id for g in scores.games if g.game_state in ("LIVE", "CRIT")]
