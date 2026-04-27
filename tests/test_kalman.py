import numpy as np
import pandas as pd
import pytest
from preprocessing.kalman_filter import TeamKalmanFilter, smooth_team_metrics


def test_filter_output_length():
    kf = TeamKalmanFilter()
    obs = np.array([10.0, 12.0, 9.0, 11.0, 13.0])
    result = kf.filter_series(obs)
    assert len(result) == len(obs)


def test_filter_smooths_spike():
    kf = TeamKalmanFilter(process_noise=1.0, measurement_noise=5.0)
    obs = np.array([10.0, 10.0, 50.0, 10.0, 10.0])
    result = kf.filter_series(obs)
    assert result[2] < 50.0, "Spike should be smoothed down"


def test_smooth_team_metrics_adds_columns():
    df = pd.DataFrame({
        "TEAM_ID": [1, 1, 1, 2, 2, 2],
        "GAME_DATE": pd.date_range("2024-01-01", periods=6),
        "NET_RATING": [5.0, 6.0, 4.0, -1.0, -2.0, 0.0],
    })
    result = smooth_team_metrics(df, metrics=["NET_RATING"])
    assert "NET_RATING_KF" in result.columns


def test_smooth_handles_missing_metric():
    df = pd.DataFrame({
        "TEAM_ID": [1, 1],
        "GAME_DATE": pd.date_range("2024-01-01", periods=2),
        "NET_RATING": [5.0, 6.0],
    })
    result = smooth_team_metrics(df, metrics=["NONEXISTENT"])
    assert "NONEXISTENT_KF" not in result.columns
