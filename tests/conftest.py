"""
Shared fixtures for the NBA Playoff Predictor test suite.

TDD philosophy: fixtures define the data contracts that each module must handle.
If a fixture shape changes, tests break immediately — by design.
"""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Kalman Filter fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_series():
    """Stable performance — filter should change nothing significantly."""
    return np.array([10.0, 10.0, 10.0, 10.0, 10.0])


@pytest.fixture
def noisy_series():
    """Single outlier spike that should be smoothed."""
    return np.array([10.0, 10.0, 60.0, 10.0, 10.0])


@pytest.fixture
def trending_series():
    """Monotonically improving team — filter should track the trend."""
    return np.array([5.0, 7.0, 9.0, 11.0, 13.0])


@pytest.fixture
def team_game_log():
    """Minimal game log DataFrame for smooth_team_metrics."""
    return pd.DataFrame({
        "TEAM_ID": [1, 1, 1, 2, 2, 2],
        "GAME_DATE": pd.date_range("2024-01-01", periods=6),
        "NET_RATING": [5.0, 6.0, 4.0, -1.0, -2.0, 0.0],
        "OFF_RATING": [110.0, 112.0, 108.0, 105.0, 104.0, 107.0],
        "DEF_RATING": [105.0, 106.0, 104.0, 106.0, 106.0, 107.0],
    })


@pytest.fixture
def game_log_with_missing(team_game_log):
    """Game log with NaN values — modules must handle missing data."""
    df = team_game_log.copy()
    df.loc[1, "NET_RATING"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Monte Carlo fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def four_teams():
    return ["Lakers", "Celtics", "Warriors", "Heat"]


@pytest.fixture
def eight_teams():
    return ["Lakers", "Celtics", "Warriors", "Heat", "Nuggets", "Bucks", "Suns", "Nets"]


@pytest.fixture
def balanced_win_probs(four_teams):
    """All matchups exactly 50/50."""
    probs = {}
    teams = four_teams
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            probs[(a, b)] = 0.5
    return probs


@pytest.fixture
def skewed_win_probs(four_teams):
    """Lakers heavily favoured in every matchup."""
    teams = four_teams
    probs = {}
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            probs[(a, b)] = 0.9 if a == "Lakers" else 0.5
    return probs


# ---------------------------------------------------------------------------
# NLP / Sentiment fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def positive_article():
    return {
        "title": "Lakers dominate Warriors in a stunning performance",
        "content": "LeBron James was unstoppable, leading LA to a convincing victory.",
        "credibility": 1.0,
        "source": "theathletic.com",
    }


@pytest.fixture
def negative_article():
    return {
        "title": "Celtics struggle in embarrassing loss",
        "content": "Boston played terribly and lost by 30 in a poor performance.",
        "credibility": 0.9,
        "source": "espn.com",
    }


@pytest.fixture
def injury_article():
    return {
        "title": "Stephen Curry questionable for Game 3",
        "content": "Warriors star is listed as doubtful with an ankle injury and may be out.",
        "credibility": 1.0,
        "source": "theathletic.com",
    }


@pytest.fixture
def low_credibility_article():
    return {
        "title": "Lakers are the GOAT team ever!!!",
        "content": "trust me bro they will win",
        "credibility": 0.3,
        "source": "reddit.com",
    }


@pytest.fixture
def news_dataframe(positive_article, negative_article, injury_article, low_credibility_article):
    rows = [positive_article, negative_article, injury_article, low_credibility_article]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stacking Classifier fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def training_dataframe():
    """Minimal labeled dataset for classifier training/evaluation."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "NET_RATING_KF": np.random.randn(n) * 3,
        "OFF_RATING_KF": np.random.randn(n) * 2 + 108,
        "DEF_RATING_KF": np.random.randn(n) * 2 + 108,
        "PACE_KF": np.random.randn(n) + 100,
        "TS_PCT_KF": np.random.uniform(0.5, 0.65, n),
        "PIE_KF": np.random.uniform(0.4, 0.6, n),
        "W_PCT_KF": np.random.uniform(0.3, 0.7, n),
        "REST_DAYS": np.random.randint(1, 5, n),
        "HOME_AWAY": np.random.randint(0, 2, n),
        "SENTIMENT_SCORE": np.random.uniform(-0.5, 0.5, n),
        "NET_RATING_DIFF": np.random.randn(n) * 3,
        "OFF_RATING_DIFF": np.random.randn(n) * 2,
        "DEF_RATING_DIFF": np.random.randn(n) * 2,
        "REST_DAYS_DIFF": np.random.randint(-3, 4, n),
        "HOME_WIN": np.random.randint(0, 2, n),
    })
