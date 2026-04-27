import pytest
from models.monte_carlo import simulate_series, run_playoff_simulation, results_to_dataframe


TEAMS = ["Lakers", "Celtics", "Warriors", "Heat"]


def test_simulate_series_winner_is_valid():
    result = simulate_series("Lakers", "Celtics", win_prob_a=0.6)
    assert result.winner in ["Lakers", "Celtics"]
    assert result.loser in ["Lakers", "Celtics"]
    assert result.winner != result.loser


def test_simulate_series_game_count():
    result = simulate_series("A", "B", win_prob_a=0.5, series_format=7)
    assert 4 <= result.games <= 7


def test_certain_win():
    wins = sum(
        simulate_series("A", "B", win_prob_a=1.0).winner == "A"
        for _ in range(100)
    )
    assert wins == 100


def test_championship_odds_sum_to_one():
    win_probs = {("Lakers", "Celtics"): 0.55, ("Warriors", "Heat"): 0.6,
                 ("Lakers", "Warriors"): 0.5, ("Lakers", "Heat"): 0.52,
                 ("Celtics", "Warriors"): 0.48, ("Celtics", "Heat"): 0.51}
    results = run_playoff_simulation(TEAMS, win_probs, n_simulations=500)
    total = sum(results.championship_odds.values())
    assert abs(total - 1.0) < 0.01


def test_results_to_dataframe():
    win_probs = {("Lakers", "Celtics"): 0.55, ("Warriors", "Heat"): 0.6,
                 ("Lakers", "Warriors"): 0.5, ("Lakers", "Heat"): 0.52,
                 ("Celtics", "Warriors"): 0.48, ("Celtics", "Heat"): 0.51}
    results = run_playoff_simulation(TEAMS, win_probs, n_simulations=200)
    df = results_to_dataframe(results)
    assert "team" in df.columns
    assert "championship_probability" in df.columns
    assert len(df) == len(TEAMS)
