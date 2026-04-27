# NBA Playoff Predictor

An end-to-end machine learning pipeline for predicting NBA playoff outcomes using quantitative signal processing, ensemble classification, and Monte Carlo simulation.

## Pipeline Overview

```
   Data Source 1                        Data Source 2
   NBA API                              NewsAPI + Reddit
   (stats, ratings, matchups)           (analyst articles, game reports)
        │                                      │
        └──────────────┬────────────────────────┘
                       ↓
              Kalman Filter
       (cleans noisy game-to-game data →
        extracts true performance signal)
                       ↓
        ┌──────────────────────────────────┐
        │     Three-Layer Stacking         │
        │  ─────────────────────────────   │
        │  Layer 1A: XGBoost               │  ← runs in parallel
        │  (gradient boosting classifier)  │
        │                                  │
        │  Layer 1B: Random Forest         │  ← runs in parallel
        │  (bagging ensemble classifier)   │
        │                                  │
        │  Layer 2:  Logistic Regression   │  ← meta-learner
        │  (combines Layer 1A + 1B output) │
        │                                  │
        │  NLP Sentiment fed in alongside  │
        │  stats as an additional feature  │
        └──────────────────────────────────┘
                       ↓
          Monte Carlo Simulation
      (10,000 bracket simulations →
       championship odds per team)
                       ↓
         Streamlit Dashboard
  (live standings, bracket odds, matchup
   predictor, sentiment heatmap, alerts)
```

---

## How the Three-Algorithm Stack Works

This project uses a **stacking classifier** — a two-level ensemble where multiple models work together rather than a single model making all decisions.

**Layer 1 — Two base learners running in parallel:**

| Algorithm | Role | Why it's here |
|-----------|------|--------------|
| **XGBoost** | Gradient boosting — builds trees sequentially, each one correcting the errors of the previous | Strong at capturing non-linear relationships in tabular stats data |
| **Random Forest** | Bagging — builds hundreds of trees independently and averages them | Reduces variance; fails differently to XGBoost so the combination is more robust |

XGBoost and Random Forest are run on the same input data **simultaneously**. Each produces its own win probability estimate.

**Layer 2 — Meta-learner combining both:**

| Algorithm | Role |
|-----------|------|
| **Logistic Regression** | Takes the outputs of XGBoost AND Random Forest as its inputs and learns the optimal way to combine them into one final prediction |

Logistic Regression is used as the meta-learner because it is interpretable — you can see exactly how much weight it gives to each base model's prediction. This mirrors how risk models at financial institutions are required to be explainable.

**Why three instead of one?**
A single algorithm has blind spots. XGBoost can overfit; Random Forest can underfit rare playoff matchup patterns. Stacking lets each model compensate for the other's weaknesses, with Logistic Regression learning which to trust more in which situations.

---

## Features

### Data Layer
- **NBA API** — RPM/RAPM proxies, True Shooting %, Usage Rate, Net/Off/Def Rating, BPM, VORP, PIE, On/Off splits, rest days
- **NewsAPI + Reddit (PRAW)** — analyst articles and r/nba posts filtered and weighted by source credibility score

### Signal Processing
- **Kalman Filter** — smooths noisy per-game stats into a stable estimate of true team/player performance level; same technique used in algorithmic trading to extract signal from price noise

### Prediction Engine
- **XGBoost** (base learner 1) — gradient boosting on Kalman-filtered stats + sentiment features
- **Random Forest** (base learner 2) — parallel bagging ensemble on the same features
- **Logistic Regression** (meta-learner) — combines XGBoost and Random Forest predictions into final win probability
- **NLP Sentiment** — VADER sentiment on credibility-weighted news sources; injected as a feature into both base learners

### Bracket Simulation
- **Monte Carlo** — simulates the full playoff bracket 10,000+ times using the stacked classifier's win probabilities; outputs championship odds, finals probabilities, and expected series lengths per matchup

### Dashboard (Streamlit)
- Live standings with Kalman-smoothed net ratings
- Championship probability bar charts updated after each game
- Credibility-weighted sentiment heatmap per team
- Injury signal alerts from news sources
- Head-to-head matchup predictor with win probability gauges

## Quantitative Methods Used

| Method | Domain Parallel |
|--------|----------------|
| Kalman Filter | Signal extraction in algorithmic trading |
| XGBoost | Credit risk / default probability modeling |
| Random Forest | Ensemble risk classification |
| Logistic Regression (meta) | Interpretable scoring model |
| Monte Carlo Simulation | Value-at-Risk, options pricing, stress testing |
| Credibility-weighted NLP | Analyst sentiment weighting in equity research |

## CI/CD

![CI](https://github.com/Hemanshu2005/nba-playoff-predictor/actions/workflows/test.yml/badge.svg)

Every push to `master` triggers a GitHub Actions pipeline that runs the full pytest suite across Kalman Filter, Monte Carlo, and NLP sentiment modules.

```
push → GitHub Actions → pytest tests/ → pass/fail badge
```

## Docker

Run the full pipeline and dashboard in a single command — no local Python setup required:

```bash
cp .env.example .env
# Add your API keys to .env
docker compose up --build
```

Dashboard available at `http://localhost:8501`.

To roll back to a previous working version:

```bash
git checkout <commit-hash>
docker compose up --build
```

## Setup (without Docker)

```bash
git clone https://github.com/Hemanshu2005/nba-playoff-predictor.git
cd nba-playoff-predictor
pip install -r requirements.txt pytest
cp .env.example .env
# Add your NewsAPI and Reddit API keys to .env
streamlit run dashboard/app.py
```

Run tests:

```bash
pytest tests/ -v
```

## Project Structure

```
nba-playoff-predictor/
├── .github/workflows/
│   └── test.yml              # GitHub Actions CI pipeline
├── data/
│   ├── fetch_nba.py          # NBA API data ingestion
│   └── fetch_news.py         # NewsAPI + Reddit fetching with credibility weights
├── preprocessing/
│   └── kalman_filter.py      # Kalman Filter for time-series smoothing
├── models/
│   ├── stacking_classifier.py  # XGBoost + RF + LR stacking pipeline
│   └── monte_carlo.py          # Bracket simulation engine
├── nlp/
│   └── sentiment.py            # VADER sentiment with source credibility weighting
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── tests/                      # pytest test suite
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## API Keys Required

| Service | Purpose | Free Tier |
|---------|---------|-----------|
| [NewsAPI](https://newsapi.org) | NBA news articles | 100 req/day |
| [Reddit API](https://www.reddit.com/prefs/apps) | r/nba posts | Free |
| NBA API | Stats (nba.com) | No key needed |
