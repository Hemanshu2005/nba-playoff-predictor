"""
TDD tests for the NLP sentiment module.

Behavioral contracts:
  - Positive text scores positive compound; negative text scores negative
  - Credibility weight scales the final score proportionally
  - Injury keywords trigger injury signal flag
  - Team mentions are extracted correctly
  - Low-credibility sources contribute less to aggregate sentiment
  - Aggregate sentiment returns a dict keyed by team name
"""
import pandas as pd
import pytest
from nlp.sentiment import (
    score_article,
    score_all_articles,
    aggregate_team_sentiment,
    has_injury_signal,
    extract_team_mentions,
    flag_injury_alerts,
)


# ---------------------------------------------------------------------------
# Contract: score_article output
# ---------------------------------------------------------------------------

def test_positive_text_scores_positive(positive_article):
    result = score_article(positive_article["title"] + " " + positive_article["content"])
    assert result["compound"] > 0


def test_negative_text_scores_negative(negative_article):
    result = score_article(negative_article["title"] + " " + negative_article["content"])
    assert result["compound"] < 0


def test_credibility_one_equals_raw_compound():
    result = score_article("Great game", credibility=1.0)
    assert result["weighted_compound"] == pytest.approx(result["compound"])


def test_credibility_scales_weighted_score():
    high = score_article("Great performance", credibility=1.0)
    low = score_article("Great performance", credibility=0.3)
    assert high["weighted_compound"] > low["weighted_compound"]


def test_empty_string_does_not_crash():
    result = score_article("")
    assert "compound" in result


def test_none_input_does_not_crash():
    result = score_article(None)
    assert "compound" in result


def test_score_returns_all_required_keys():
    result = score_article("test text")
    for key in ["compound", "weighted_compound", "positive", "negative", "neutral", "credibility"]:
        assert key in result


# ---------------------------------------------------------------------------
# Contract: injury signal detection
# ---------------------------------------------------------------------------

def test_injury_signal_detected_questionable(injury_article):
    text = injury_article["title"] + " " + injury_article["content"]
    assert has_injury_signal(text) is True


def test_injury_signal_detected_out():
    assert has_injury_signal("LeBron James is out for game 3") is True


def test_injury_signal_detected_doubtful():
    assert has_injury_signal("Curry listed as doubtful") is True


def test_no_injury_signal_in_positive_article(positive_article):
    text = positive_article["title"] + " " + positive_article["content"]
    assert has_injury_signal(text) is False


# ---------------------------------------------------------------------------
# Contract: team mention extraction
# ---------------------------------------------------------------------------

def test_lakers_mentioned():
    mentions = extract_team_mentions("The Lakers played well tonight")
    assert "lakers" in mentions


def test_multiple_teams_extracted():
    mentions = extract_team_mentions("Lakers beat Celtics in overtime")
    assert "lakers" in mentions
    assert "celtics" in mentions


def test_no_team_mentioned_returns_empty():
    mentions = extract_team_mentions("The weather was great today")
    assert mentions == []


# ---------------------------------------------------------------------------
# Contract: score_all_articles DataFrame transformation
# ---------------------------------------------------------------------------

def test_score_all_articles_adds_sentiment_columns(news_dataframe):
    result = score_all_articles(news_dataframe)
    assert "compound_sentiment" in result.columns
    assert "weighted_sentiment" in result.columns
    assert "injury_signal" in result.columns
    assert "team_mentions" in result.columns


def test_injury_article_flagged_in_dataframe(news_dataframe):
    result = score_all_articles(news_dataframe)
    injury_rows = result[result["injury_signal"] == True]
    assert len(injury_rows) >= 1


def test_row_count_preserved(news_dataframe):
    result = score_all_articles(news_dataframe)
    assert len(result) == len(news_dataframe)


# ---------------------------------------------------------------------------
# Contract: aggregate_team_sentiment
# ---------------------------------------------------------------------------

def test_aggregate_returns_dict(news_dataframe):
    scored = score_all_articles(news_dataframe)
    result = aggregate_team_sentiment(scored)
    assert isinstance(result, dict)


def test_teams_with_no_mentions_return_zero(news_dataframe):
    scored = score_all_articles(news_dataframe)
    result = aggregate_team_sentiment(scored)
    assert result.get("spurs", 0.0) == 0.0


def test_low_credibility_contributes_less(positive_article, low_credibility_article):
    high_df = pd.DataFrame([positive_article])
    low_df = pd.DataFrame([low_credibility_article])
    high_scored = score_all_articles(high_df)
    low_scored = score_all_articles(low_df)
    high_agg = aggregate_team_sentiment(high_scored)
    low_agg = aggregate_team_sentiment(low_scored)
    lakers_high = abs(high_agg.get("lakers", 0.0))
    lakers_low = abs(low_agg.get("lakers", 0.0))
    assert lakers_high >= lakers_low


# ---------------------------------------------------------------------------
# Contract: flag_injury_alerts
# ---------------------------------------------------------------------------

def test_flag_injury_alerts_returns_only_injuries(news_dataframe):
    scored = score_all_articles(news_dataframe)
    alerts = flag_injury_alerts(scored)
    assert all(alerts["injury_signal"] == True)


def test_flag_injury_alerts_sorted_by_credibility(news_dataframe):
    scored = score_all_articles(news_dataframe)
    alerts = flag_injury_alerts(scored)
    if len(alerts) > 1:
        creds = alerts["credibility"].tolist()
        assert creds == sorted(creds, reverse=True)
