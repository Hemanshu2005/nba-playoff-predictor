"""
TDD tests for the Kalman Filter module.

Each test specifies a behavioral contract:
  - Output shape must match input shape
  - Spikes must be attenuated
  - Trends must be tracked
  - Missing data must not crash the pipeline
  - Each team is smoothed independently
"""
import numpy as np
import pandas as pd
import pytest
from preprocessing.kalman_filter import TeamKalmanFilter, smooth_team_metrics


# ---------------------------------------------------------------------------
# Contract: output shape
# ---------------------------------------------------------------------------

def test_output_length_matches_input(flat_series):
    kf = TeamKalmanFilter()
    assert len(kf.filter_series(flat_series)) == len(flat_series)


def test_single_observation_returns_single_value():
    kf = TeamKalmanFilter()
    result = kf.filter_series(np.array([7.5]))
    assert len(result) == 1
    assert result[0] == pytest.approx(7.5)


def test_output_is_numpy_array(flat_series):
    kf = TeamKalmanFilter()
    result = kf.filter_series(flat_series)
    assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Contract: noise attenuation
# ---------------------------------------------------------------------------

def test_spike_is_attenuated(noisy_series):
    kf = TeamKalmanFilter(process_noise=1.0, measurement_noise=5.0)
    result = kf.filter_series(noisy_series)
    assert result[2] < noisy_series[2], "Spike at index 2 must be smoothed down"


def test_spike_attenuation_increases_with_measurement_noise(noisy_series):
    low_noise = TeamKalmanFilter(process_noise=1.0, measurement_noise=1.0)
    high_noise = TeamKalmanFilter(process_noise=1.0, measurement_noise=20.0)
    result_low = low_noise.filter_series(noisy_series)
    result_high = high_noise.filter_series(noisy_series)
    assert result_high[2] < result_low[2], "Higher measurement noise → more spike suppression"


def test_flat_series_stays_near_constant(flat_series):
    kf = TeamKalmanFilter()
    result = kf.filter_series(flat_series)
    assert np.allclose(result, 10.0, atol=0.5)


# ---------------------------------------------------------------------------
# Contract: trend tracking
# ---------------------------------------------------------------------------

def test_filter_tracks_upward_trend(trending_series):
    kf = TeamKalmanFilter(process_noise=2.0, measurement_noise=1.0)
    result = kf.filter_series(trending_series)
    assert result[-1] > result[0], "Filter must track an improving team upward"


def test_smoothed_values_lag_behind_trend(trending_series):
    kf = TeamKalmanFilter()
    result = kf.filter_series(trending_series)
    assert result[-1] <= trending_series[-1], "Smoothed value must not overshoot raw observation"


# ---------------------------------------------------------------------------
# Contract: DataFrame integration
# ---------------------------------------------------------------------------

def test_smooth_team_metrics_adds_kf_column(team_game_log):
    result = smooth_team_metrics(team_game_log, metrics=["NET_RATING"])
    assert "NET_RATING_KF" in result.columns


def test_smooth_multiple_metrics(team_game_log):
    result = smooth_team_metrics(team_game_log, metrics=["NET_RATING", "OFF_RATING"])
    assert "NET_RATING_KF" in result.columns
    assert "OFF_RATING_KF" in result.columns


def test_unknown_metric_is_skipped(team_game_log):
    result = smooth_team_metrics(team_game_log, metrics=["NONEXISTENT"])
    assert "NONEXISTENT_KF" not in result.columns


def test_original_columns_preserved(team_game_log):
    result = smooth_team_metrics(team_game_log, metrics=["NET_RATING"])
    assert "NET_RATING" in result.columns


def test_each_team_smoothed_independently(team_game_log):
    result = smooth_team_metrics(team_game_log, metrics=["NET_RATING"])
    team1 = result[result["TEAM_ID"] == 1]["NET_RATING_KF"].values
    team2 = result[result["TEAM_ID"] == 2]["NET_RATING_KF"].values
    assert not np.allclose(team1, team2), "Teams must be smoothed separately"


def test_handles_missing_values(game_log_with_missing):
    result = smooth_team_metrics(game_log_with_missing, metrics=["NET_RATING"])
    assert result["NET_RATING_KF"].isna().sum() == 0
