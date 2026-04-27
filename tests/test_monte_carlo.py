"""
TDD tests for the Monte Carlo simulation module.

Behavioral contracts:
  - Series winner is always one of the two teams
  - Game count stays within valid bounds for series format
  - A team with win_prob=1.0 must always win
  - Championship odds across all teams must sum to 1.0
  - A heavily favoured team must win the championship more often
  - Results DataFrame has required columns and correct row count
"""
import pytest
import numpy as np
from models.monte_carlo import (
    simulate_series,
    run_playoff_simulation,
    results_to_dataframe,
    get_win_probability,
)


# ---------------------------------------------------------------------------
# Contract: simulate_series output validity
# ---------------------------------------------------------------------------

def test_winner_is_one_of_the_two_teams():
    result = simulate_series("Lakers", "Celtics", win_prob_a=0.6)
    assert result.winner in ["Lakers", "Celtics"]


def test_winner_and_loser_are_different():
    result = simulate_series("A", "B", win_prob_a=0.5)
    assert result.winner != result.loser


def test_game_count_within_bounds_best_of_7():
    for _ in range(50):
        result = simulate_series("A", "B", win_prob_a=0.5, series_format=7)
        assert 4 <= result.games <= 7


def test_game_count_within_bounds_best_of_5():
    for _ in range(50):
        result = simulate_series("A", "B", win_prob_a=0.5, series_format=5)
        assert 3 <= result.games <= 5


def test_certain_favourite_always_wins():
    for _ in range(100):
        result = simulate_series("A", "B", win_prob_a=1.0)
        assert result.winner == "A"


def test_certain_underdog_never_wins():
    for _ in range(100):
        result = simulate_series("A", "B", win_prob_a=0.0)
        assert result.winner == "B"


# ---------------------------------------------------------------------------
# Contract: run_playoff_simulation statistical properties
# ---------------------------------------------------------------------------

def test_championship_odds_sum_to_one(four_teams, balanced_win_probs):
    results = run_playoff_simulation(four_teams, balanced_win_probs, n_simulations=1000)
    total = sum(results.championship_odds.values())
    assert abs(total - 1.0) < 0.02


def test_all_teams_represented_in_odds(four_teams, balanced_win_probs):
    results = run_playoff_simulation(four_teams, balanced_win_probs, n_simulations=500)
    for team in four_teams:
        assert team in results.championship_odds


def test_balanced_probs_give_similar_odds(four_teams, balanced_win_probs):
    results = run_playoff_simulation(four_teams, balanced_win_probs, n_simulations=4000)
    odds = list(results.championship_odds.values())
    assert max(odds) - min(odds) < 0.3, "Balanced bracket should not heavily favour one team"


def test_favoured_team_wins_most(four_teams, skewed_win_probs):
    results = run_playoff_simulation(four_teams, skewed_win_probs, n_simulations=2000)
    best_team = max(results.championship_odds, key=results.championship_odds.get)
    assert best_team == "Lakers"


def test_finals_odds_geq_championship_odds(four_teams, balanced_win_probs):
    results = run_playoff_simulation(four_teams, balanced_win_probs, n_simulations=1000)
    for team in four_teams:
        assert results.finals_odds.get(team, 0) >= results.championship_odds.get(team, 0)


def test_unknown_matchup_defaults_to_fifty_fifty():
    prob = get_win_probability({}, "TeamX", "TeamY")
    assert prob == 0.5


def test_reversed_matchup_key_is_handled():
    probs = {("B", "A"): 0.7}
    prob = get_win_probability(probs, "A", "B")
    assert prob == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Contract: results_to_dataframe structure
# ---------------------------------------------------------------------------

def test_dataframe_has_required_columns(four_teams, balanced_win_probs):
    results = run_playoff_simulation(four_teams, balanced_win_probs, n_simulations=200)
    df = results_to_dataframe(results)
    assert "team" in df.columns
    assert "championship_probability" in df.columns
    assert "finals_probability" in df.columns


def test_dataframe_row_count_matches_teams(four_teams, balanced_win_probs):
    results = run_playoff_simulation(four_teams, balanced_win_probs, n_simulations=200)
    df = results_to_dataframe(results)
    assert len(df) == len(four_teams)


def test_dataframe_sorted_by_championship_probability(four_teams, skewed_win_probs):
    results = run_playoff_simulation(four_teams, skewed_win_probs, n_simulations=500)
    df = results_to_dataframe(results)
    probs = df["championship_probability"].tolist()
    assert probs == sorted(probs, reverse=True)
