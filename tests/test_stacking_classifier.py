"""
TDD tests for the Stacking Classifier module.

Behavioral contracts:
  - Pipeline trains without error on valid data
  - Predictions are binary (0 or 1)
  - Probabilities are in [0, 1] and sum to 1 per row
  - Evaluation metrics are within valid ranges
  - predict_matchup returns home + away probabilities summing to 1
  - Missing features are handled without crashing
"""
import numpy as np
import pandas as pd
import pytest
from models.stacking_classifier import (
    build_pipeline,
    prepare_features,
    train,
    evaluate,
    predict_matchup,
)


# ---------------------------------------------------------------------------
# Contract: feature preparation
# ---------------------------------------------------------------------------

def test_prepare_features_drops_unknown_columns(training_dataframe):
    df = training_dataframe.copy()
    df["UNKNOWN_COL"] = 999
    X, y = prepare_features(df)
    assert "UNKNOWN_COL" not in X.columns


def test_prepare_features_returns_target_series(training_dataframe):
    X, y = prepare_features(training_dataframe)
    assert y is not None
    assert len(y) == len(training_dataframe)


def test_prepare_features_no_target_returns_none():
    df = pd.DataFrame({"NET_RATING_KF": [1.0, 2.0], "REST_DAYS": [2, 3]})
    X, y = prepare_features(df)
    assert y is None


def test_prepare_features_fills_missing_values(training_dataframe):
    df = training_dataframe.copy()
    df.loc[0, "NET_RATING_KF"] = np.nan
    X, y = prepare_features(df)
    assert X.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# Contract: training
# ---------------------------------------------------------------------------

def test_pipeline_trains_without_error(training_dataframe):
    pipeline = train(training_dataframe)
    assert pipeline is not None


def test_pipeline_has_scaler_and_model(training_dataframe):
    pipeline = train(training_dataframe)
    assert "scaler" in pipeline.named_steps
    assert "model" in pipeline.named_steps


# ---------------------------------------------------------------------------
# Contract: prediction output
# ---------------------------------------------------------------------------

def test_predictions_are_binary(training_dataframe):
    pipeline = train(training_dataframe)
    X, _ = prepare_features(training_dataframe)
    preds = pipeline.predict(X)
    assert set(preds).issubset({0, 1})


def test_probabilities_in_valid_range(training_dataframe):
    pipeline = train(training_dataframe)
    X, _ = prepare_features(training_dataframe)
    proba = pipeline.predict_proba(X)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_probabilities_sum_to_one(training_dataframe):
    pipeline = train(training_dataframe)
    X, _ = prepare_features(training_dataframe)
    proba = pipeline.predict_proba(X)
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Contract: evaluation metrics
# ---------------------------------------------------------------------------

def test_accuracy_between_zero_and_one(training_dataframe):
    pipeline = train(training_dataframe)
    metrics = evaluate(pipeline, training_dataframe)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_roc_auc_between_zero_and_one(training_dataframe):
    pipeline = train(training_dataframe)
    metrics = evaluate(pipeline, training_dataframe)
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_log_loss_is_positive(training_dataframe):
    pipeline = train(training_dataframe)
    metrics = evaluate(pipeline, training_dataframe)
    assert metrics["log_loss"] >= 0.0


def test_evaluate_returns_all_metric_keys(training_dataframe):
    pipeline = train(training_dataframe)
    metrics = evaluate(pipeline, training_dataframe)
    for key in ["accuracy", "log_loss", "roc_auc"]:
        assert key in metrics


# ---------------------------------------------------------------------------
# Contract: predict_matchup
# ---------------------------------------------------------------------------

def test_predict_matchup_probabilities_sum_to_one(training_dataframe):
    pipeline = train(training_dataframe)
    home_features = {"NET_RATING_KF": 5.0, "OFF_RATING_KF": 112.0, "REST_DAYS": 2}
    away_features = {"NET_RATING_KF": 2.0, "OFF_RATING_KF": 108.0, "REST_DAYS": 1}
    result = predict_matchup(pipeline, home_features, away_features, sentiment_score=0.1)
    total = result["home_win_probability"] + result["away_win_probability"]
    assert abs(total - 1.0) < 1e-4


def test_predict_matchup_returns_required_keys(training_dataframe):
    pipeline = train(training_dataframe)
    result = predict_matchup(pipeline, {}, {})
    assert "home_win_probability" in result
    assert "away_win_probability" in result


def test_better_team_has_higher_win_prob(training_dataframe):
    pipeline = train(training_dataframe)
    strong = {"NET_RATING_KF": 10.0, "W_PCT_KF": 0.75}
    weak = {"NET_RATING_KF": -5.0, "W_PCT_KF": 0.3}
    result = predict_matchup(pipeline, strong, weak, sentiment_score=0.0)
    assert result["home_win_probability"] > result["away_win_probability"]
