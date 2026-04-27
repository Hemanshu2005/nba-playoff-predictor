# NBA Playoff Predictor

An end-to-end machine learning pipeline for predicting NBA playoff outcomes using quantitative signal processing, ensemble classification, and Monte Carlo simulation.

## Pipeline Overview

```
NBA API + News/Reddit
        ↓
   Kalman Filter          — smooths noisy game-to-game stats into true performance signal
        ↓
  Stacking Classifier     — XGBoost + Random Forest base learners, Logistic Regression meta
        ↓  (NLP sentiment fed in parallel)
  Monte Carlo Simulation  — 10,000 bracket iterations → championship odds with confidence intervals
        ↓
  Streamlit Dashboard     — live standings, bracket odds, matchup predictor, injury alerts
```

## Features

### Data Layer
- **NBA API** — RPM/RAPM proxies, True Shooting %, Usage Rate, Net/Off/Def Rating, BPM, VORP, PIE, On/Off splits, rest days
- **NewsAPI + Reddit (PRAW)** — analyst articles and r/nba posts filtered and weighted by source credibility

### Signal Processing
- **Kalman Filter** — extracts true team/player performance level from noisy game observations; same technique used in algorithmic trading to separate signal from price noise

### Prediction Engine
- **Stacking Classifier**
  - Base learner 1: **XGBoost** — gradient boosting with sequential error correction
  - Base learner 2: **Random Forest** — bagging with parallel variance reduction
  - Meta-learner: **Logistic Regression** — interpretable combination of base predictions
- **NLP Sentiment** — VADER sentiment scored on credibility-weighted news sources; fed as a feature alongside stats

### Bracket Simulation
- **Monte Carlo** — simulates the full playoff bracket 10,000+ times using per-matchup win probabilities from the stacking classifier; outputs championship odds, finals probabilities, and expected series lengths

### Dashboard (Streamlit)
- Live standings with Kalman-smoothed net ratings
- Championship probability bar charts updated after each game
- Credibility-weighted sentiment heatmap per team
- Injury signal alerts from news
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
