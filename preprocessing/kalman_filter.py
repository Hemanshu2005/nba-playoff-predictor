import numpy as np
import pandas as pd
from typing import List


class TeamKalmanFilter:
    """
    1-D Kalman Filter applied per team per metric.

    Models a team's "true" performance level as a hidden state that
    evolves over time. Measurement noise accounts for game-to-game
    variance (blowouts, back-to-backs, garbage time). This is the same
    filter used in algorithmic trading to extract signal from noisy
    price series.

    State:   x  — estimated true performance level
    Process: x_{t+1} = x_t + w_t,  w_t ~ N(0, Q)   (performance drifts)
    Measure: z_t = x_t + v_t,       v_t ~ N(0, R)   (we observe noisy game stats)
    """

    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 5.0):
        self.Q = process_noise       # how much true performance can shift game-to-game
        self.R = measurement_noise   # how noisy individual game measurements are

    def filter_series(self, observations: np.ndarray) -> np.ndarray:
        n = len(observations)
        x = observations[0]   # initial state estimate
        P = 10.0              # initial uncertainty

        smoothed = np.zeros(n)
        smoothed[0] = x

        for t in range(1, n):
            # Predict
            x_pred = x
            P_pred = P + self.Q

            # Update
            K = P_pred / (P_pred + self.R)          # Kalman gain
            x = x_pred + K * (observations[t] - x_pred)
            P = (1 - K) * P_pred

            smoothed[t] = x

        return smoothed


def smooth_team_metrics(
    df: pd.DataFrame,
    metrics: List[str],
    group_col: str = "TEAM_ID",
    date_col: str = "GAME_DATE",
    process_noise: float = 1.0,
    measurement_noise: float = 5.0,
) -> pd.DataFrame:
    """
    Apply Kalman filtering to each metric for each team independently.
    Returns the original dataframe with smoothed columns appended (_KF suffix).
    """
    kf = TeamKalmanFilter(process_noise=process_noise, measurement_noise=measurement_noise)
    df = df.copy().sort_values([group_col, date_col])

    for metric in metrics:
        if metric not in df.columns:
            continue

        smoothed_col = f"{metric}_KF"
        df[smoothed_col] = np.nan

        for team_id, group in df.groupby(group_col):
            obs = group[metric].fillna(group[metric].mean()).values
            smoothed = kf.filter_series(obs)
            df.loc[group.index, smoothed_col] = smoothed

    return df


def smooth_player_metrics(
    df: pd.DataFrame,
    metrics: List[str],
    group_col: str = "PLAYER_ID",
    date_col: str = "GAME_DATE",
    process_noise: float = 0.5,
    measurement_noise: float = 4.0,
) -> pd.DataFrame:
    """Same as smooth_team_metrics but scoped to individual players."""
    return smooth_team_metrics(
        df, metrics,
        group_col=group_col,
        date_col=date_col,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
    )


TEAM_METRICS_TO_SMOOTH = [
    "NET_RATING", "OFF_RATING", "DEF_RATING",
    "PACE", "TS_PCT", "PIE",
    "W_PCT",
]

PLAYER_METRICS_TO_SMOOTH = [
    "NET_RATING", "OFF_RATING", "DEF_RATING",
    "TS_PCT", "USG_PCT", "PIE",
    "PTS", "REB", "AST", "PLUS_MINUS",
]
