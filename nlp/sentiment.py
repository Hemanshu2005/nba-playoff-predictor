import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


analyzer = SentimentIntensityAnalyzer()

NBA_TEAMS = [
    "lakers", "celtics", "warriors", "heat", "bucks", "nuggets", "suns",
    "clippers", "nets", "76ers", "sixers", "raptors", "bulls", "knicks",
    "mavericks", "mavs", "grizzlies", "pelicans", "thunder", "jazz",
    "timberwolves", "wolves", "rockets", "spurs", "kings", "pacers",
    "cavaliers", "cavs", "hawks", "hornets", "magic", "pistons", "blazers",
    "wizards",
]

INJURY_KEYWORDS = ["out", "doubtful", "questionable", "injured", "sidelined", "dnp"]
HYPE_KEYWORDS = ["dominant", "unstoppable", "elite", "MVP", "best", "locked in"]


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", "", text)
    return text.strip().lower()


def extract_team_mentions(text: str) -> List[str]:
    text = text.lower()
    return [team for team in NBA_TEAMS if team in text]


def score_article(text: str, credibility: float = 1.0) -> Dict[str, float]:
    """
    Returns a credibility-weighted compound sentiment score.
    VADER compound ranges from -1 (very negative) to +1 (very positive).
    """
    cleaned = clean_text(text)
    scores = analyzer.polarity_scores(cleaned)
    weighted = scores["compound"] * credibility
    return {
        "compound": scores["compound"],
        "weighted_compound": weighted,
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "credibility": credibility,
    }


def has_injury_signal(text: str) -> bool:
    text = text.lower()
    return any(kw in text for kw in INJURY_KEYWORDS)


def score_all_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: title, content, credibility, source.
    Returns dataframe with sentiment columns added.
    """
    df = df.copy()
    texts = (df["title"].fillna("") + " " + df["content"].fillna("")).tolist()
    credibilities = df["credibility"].tolist()

    rows = [score_article(t, c) for t, c in zip(texts, credibilities)]
    sentiment_df = pd.DataFrame(rows)

    df["compound_sentiment"] = sentiment_df["compound"].values
    df["weighted_sentiment"] = sentiment_df["weighted_compound"].values
    df["injury_signal"] = [has_injury_signal(t) for t in texts]
    df["team_mentions"] = [extract_team_mentions(t) for t in texts]

    return df


def aggregate_team_sentiment(scored_df: pd.DataFrame) -> Dict[str, float]:
    """
    Returns a dict mapping team name → credibility-weighted average sentiment.
    Used as a feature input to the stacking classifier.
    """
    team_scores: Dict[str, List[float]] = {team: [] for team in NBA_TEAMS}

    for _, row in scored_df.iterrows():
        for team in row.get("team_mentions", []):
            team_scores[team].append(row["weighted_sentiment"])

    return {
        team: float(np.mean(scores)) if scores else 0.0
        for team, scores in team_scores.items()
    }


def get_matchup_sentiment(
    team_a: str,
    team_b: str,
    team_sentiments: Dict[str, float],
) -> float:
    """
    Returns net sentiment differential: team_a score minus team_b score.
    Positive = media favours team_a; negative = media favours team_b.
    """
    score_a = team_sentiments.get(team_a.lower(), 0.0)
    score_b = team_sentiments.get(team_b.lower(), 0.0)
    return round(score_a - score_b, 4)


def flag_injury_alerts(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Returns only articles containing injury signals, sorted by credibility."""
    return (
        scored_df[scored_df["injury_signal"]]
        .sort_values("credibility", ascending=False)
        .reset_index(drop=True)
    )
