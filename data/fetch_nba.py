import pandas as pd
from nba_api.stats.endpoints import (
    leaguegamelog,
    playergamelogs,
    teamestimatedmetrics,
    leaguedashplayerstats,
    leaguedashteamstats,
)
from nba_api.stats.static import teams, players
import time


CURRENT_SEASON = "2024-25"


def get_all_teams() -> pd.DataFrame:
    nba_teams = teams.get_teams()
    return pd.DataFrame(nba_teams)


def get_team_game_logs(season: str = CURRENT_SEASON) -> pd.DataFrame:
    logs = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Playoffs",
        player_or_team_abbreviation="T",
    )
    df = logs.get_data_frames()[0]
    return df


def get_player_advanced_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """
    Pulls RPM-adjacent metrics: BPM, VORP, PER, TS%, USG%, Net Rating,
    Offensive/Defensive Rating, On/Off splits.
    """
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_simple="Advanced",
        per_mode_simple="PerGame",
    )
    df = stats.get_data_frames()[0]
    cols = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
        "NET_RATING", "OFF_RATING", "DEF_RATING",
        "TS_PCT", "USG_PCT", "PIE",
        "AST_PCT", "REB_PCT",
        "GP", "MIN",
    ]
    return df[[c for c in cols if c in df.columns]]


def get_team_advanced_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_simple="Advanced",
        per_mode_simple="PerGame",
    )
    df = stats.get_data_frames()[0]
    cols = [
        "TEAM_ID", "TEAM_NAME",
        "NET_RATING", "OFF_RATING", "DEF_RATING",
        "PACE", "PIE", "TS_PCT",
        "GP", "W", "L",
    ]
    return df[[c for c in cols if c in df.columns]]


def get_estimated_metrics(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """RAPM-proxy: estimated on/off metrics from NBA.com."""
    metrics = teamestimatedmetrics.TeamEstimatedMetrics(season=season)
    df = metrics.get_data_frames()[0]
    return df


def get_player_game_logs(season: str = CURRENT_SEASON) -> pd.DataFrame:
    logs = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable="Playoffs",
    )
    df = logs.get_data_frames()[0]
    time.sleep(0.6)
    return df


def get_rest_days(game_logs: pd.DataFrame) -> pd.DataFrame:
    """Compute rest days between games per team."""
    df = game_logs.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    df["REST_DAYS"] = df.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days.fillna(3)
    return df


def get_injury_report() -> pd.DataFrame:
    """
    Placeholder — NBA.com does not expose a public injury API.
    Populate via manual CSV or a third-party scraper.
    """
    try:
        return pd.read_csv("data/injury_report.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["PLAYER_NAME", "TEAM_ABBREVIATION", "STATUS", "DATE"])
