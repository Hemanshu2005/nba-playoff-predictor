import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


SIMULATIONS = 10_000

PLAYOFF_BRACKET = {
    "East": {
        "R1": [("1E", "8E"), ("4E", "5E"), ("3E", "6E"), ("2E", "7E")],
        "R2": [("W1E_8E", "W4E_5E"), ("W3E_6E", "W2E_7E")],
        "CF": [("WR2_1", "WR2_2")],
    },
    "West": {
        "R1": [("1W", "8W"), ("4W", "5W"), ("3W", "6W"), ("2W", "7W")],
        "R2": [("W1W_8W", "W4W_5W"), ("W3W_6W", "W2W_7W")],
        "CF": [("WR2_3", "WR2_4")],
    },
    "Finals": [("ECF_Winner", "WCF_Winner")],
}


@dataclass
class SeriesResult:
    winner: str
    loser: str
    games: int
    win_prob: float


@dataclass
class SimulationResults:
    championship_odds: Dict[str, float] = field(default_factory=dict)
    finals_odds: Dict[str, float] = field(default_factory=dict)
    conference_finals_odds: Dict[str, float] = field(default_factory=dict)
    round2_odds: Dict[str, float] = field(default_factory=dict)
    avg_series_length: Dict[Tuple[str, str], float] = field(default_factory=dict)


def simulate_game(win_prob_team_a: float) -> bool:
    """Returns True if team A wins."""
    return np.random.random() < win_prob_team_a


def simulate_series(
    team_a: str,
    team_b: str,
    win_prob_a: float,
    series_format: int = 7,
) -> SeriesResult:
    """
    Simulates a best-of-N series.
    Returns winner, loser, and total games played.
    """
    wins_needed = (series_format // 2) + 1
    wins_a = wins_b = 0

    while wins_a < wins_needed and wins_b < wins_needed:
        if simulate_game(win_prob_a):
            wins_a += 1
        else:
            wins_b += 1

    winner = team_a if wins_a == wins_needed else team_b
    loser = team_b if winner == team_a else team_a
    return SeriesResult(
        winner=winner,
        loser=loser,
        games=wins_a + wins_b,
        win_prob=win_prob_a,
    )


def run_playoff_simulation(
    teams: List[str],
    win_probs: Dict[Tuple[str, str], float],
    n_simulations: int = SIMULATIONS,
) -> SimulationResults:
    """
    Runs n_simulations full playoff brackets.

    win_probs: dict mapping (team_a, team_b) → probability team_a wins a single game.
               Probabilities come from the stacking classifier output.
    """
    results = SimulationResults()
    champ_counts: Dict[str, int] = {t: 0 for t in teams}
    finals_counts: Dict[str, int] = {t: 0 for t in teams}
    series_game_totals: Dict[Tuple[str, str], List[int]] = {}

    for _ in range(n_simulations):
        _simulate_bracket(
            teams, win_probs,
            champ_counts, finals_counts, series_game_totals,
        )

    results.championship_odds = {
        t: round(c / n_simulations, 4) for t, c in champ_counts.items()
    }
    results.finals_odds = {
        t: round(c / n_simulations, 4) for t, c in finals_counts.items()
    }
    results.avg_series_length = {
        matchup: round(float(np.mean(lengths)), 2)
        for matchup, lengths in series_game_totals.items()
    }

    return results


def _simulate_bracket(
    teams: List[str],
    win_probs: Dict[Tuple[str, str], float],
    champ_counts: Dict[str, int],
    finals_counts: Dict[str, int],
    series_game_totals: Dict[Tuple[str, str], List[int]],
) -> None:
    """Single bracket simulation — mutates counters in place."""
    remaining = list(teams)

    while len(remaining) > 1:
        next_round = []
        np.random.shuffle(remaining)

        for i in range(0, len(remaining), 2):
            if i + 1 >= len(remaining):
                next_round.append(remaining[i])
                continue

            a, b = remaining[i], remaining[i + 1]
            prob = win_probs.get((a, b), win_probs.get((b, a), 0.5))
            if (b, a) in win_probs and (a, b) not in win_probs:
                prob = 1 - prob

            result = simulate_series(a, b, prob)
            key = tuple(sorted([a, b]))
            series_game_totals.setdefault(key, []).append(result.games)
            next_round.append(result.winner)

        if len(remaining) == 2:
            for t in remaining:
                finals_counts[t] += 1

        remaining = next_round

    champ_counts[remaining[0]] += 1


def get_win_probability(
    matchup_probs: Dict[Tuple[str, str], float],
    team_a: str,
    team_b: str,
) -> float:
    if (team_a, team_b) in matchup_probs:
        return matchup_probs[(team_a, team_b)]
    if (team_b, team_a) in matchup_probs:
        return 1 - matchup_probs[(team_b, team_a)]
    return 0.5


def results_to_dataframe(results: SimulationResults) -> pd.DataFrame:
    rows = []
    for team, champ_prob in results.championship_odds.items():
        rows.append({
            "team": team,
            "championship_probability": champ_prob,
            "finals_probability": results.finals_odds.get(team, 0.0),
        })
    return pd.DataFrame(rows).sort_values("championship_probability", ascending=False).reset_index(drop=True)
