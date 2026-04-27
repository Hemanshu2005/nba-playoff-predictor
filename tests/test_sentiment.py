import pandas as pd
import pytest
from nlp.sentiment import score_article, score_all_articles, aggregate_team_sentiment, has_injury_signal


def test_positive_sentiment():
    result = score_article("The Lakers looked dominant and unstoppable tonight.", credibility=1.0)
    assert result["compound"] > 0


def test_negative_sentiment():
    result = score_article("The team played terribly and lost badly.", credibility=1.0)
    assert result["compound"] < 0


def test_credibility_scales_score():
    high = score_article("Great performance", credibility=1.0)
    low = score_article("Great performance", credibility=0.3)
    assert high["weighted_compound"] > low["weighted_compound"]


def test_injury_signal_detected():
    assert has_injury_signal("LeBron James is questionable for game 3") is True
    assert has_injury_signal("Lakers win in dominant fashion") is False


def test_score_all_articles():
    df = pd.DataFrame({
        "title": ["Lakers dominate", "Celtics injury concern"],
        "content": ["Great win for LA", "Jayson Tatum is out for game 2"],
        "credibility": [0.9, 1.0],
        "source": ["espn", "theathletic"],
    })
    result = score_all_articles(df)
    assert "weighted_sentiment" in result.columns
    assert "injury_signal" in result.columns
    assert result["injury_signal"].iloc[1] is True


def test_aggregate_team_sentiment_returns_dict():
    df = pd.DataFrame({
        "title": ["Lakers look great"],
        "content": ["The lakers are playing well"],
        "credibility": [1.0],
        "source": ["espn"],
    })
    scored = score_all_articles(df)
    result = aggregate_team_sentiment(scored)
    assert isinstance(result, dict)
    assert "lakers" in result
