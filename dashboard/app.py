import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data.fetch_nba import get_team_advanced_stats, get_team_game_logs, get_rest_days
from data.fetch_news import load_all_news
from nlp.sentiment import score_all_articles, aggregate_team_sentiment, flag_injury_alerts
from models.monte_carlo import run_playoff_simulation, results_to_dataframe
from preprocessing.kalman_filter import smooth_team_metrics, TEAM_METRICS_TO_SMOOTH


st.set_page_config(
    page_title="NBA Playoff Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏀 NBA Playoff Predictor")
st.caption("Kalman-filtered stats · XGBoost + Random Forest stacking · Monte Carlo bracket simulation · Credibility-weighted sentiment")


@st.cache_data(ttl=3600)
def load_data():
    team_stats = get_team_advanced_stats()
    game_logs = get_team_game_logs()
    game_logs = get_rest_days(game_logs)
    news_df = load_all_news("NBA playoffs")
    scored_news = score_all_articles(news_df)
    team_sentiment = aggregate_team_sentiment(scored_news)
    return team_stats, game_logs, scored_news, team_sentiment


@st.cache_data(ttl=3600)
def load_smoothed_stats():
    game_logs = get_team_game_logs()
    game_logs["GAME_DATE"] = pd.to_datetime(game_logs["GAME_DATE"])
    smoothed = smooth_team_metrics(
        game_logs,
        metrics=["PLUS_MINUS"],
        group_col="TEAM_ID",
        date_col="GAME_DATE",
    )
    return smoothed


with st.spinner("Loading live data..."):
    try:
        team_stats, game_logs, scored_news, team_sentiment = load_data()
        data_loaded = True
    except Exception as e:
        st.warning(f"Live data unavailable: {e}. Showing demo mode.")
        data_loaded = False
        team_stats = pd.DataFrame()
        team_sentiment = {}
        scored_news = pd.DataFrame()


tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Standings & Stats",
    "🏆 Bracket Odds",
    "📰 Sentiment & News",
    "⚔️ Matchup Predictor",
])


with tab1:
    st.subheader("Current Team Standings")

    if data_loaded and not team_stats.empty:
        display_cols = [c for c in [
            "TEAM_NAME", "W", "L", "NET_RATING",
            "OFF_RATING", "DEF_RATING", "PACE", "TS_PCT",
        ] if c in team_stats.columns]

        styled = team_stats[display_cols].copy()
        if "NET_RATING" in styled.columns:
            styled = styled.sort_values("NET_RATING", ascending=False).reset_index(drop=True)

        st.dataframe(styled, use_container_width=True)

        if "NET_RATING" in team_stats.columns and "TEAM_NAME" in team_stats.columns:
            fig = px.bar(
                team_stats.sort_values("NET_RATING", ascending=True).tail(15),
                x="NET_RATING", y="TEAM_NAME",
                orientation="h",
                color="NET_RATING",
                color_continuous_scale="RdYlGn",
                title="Net Rating by Team",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Connect NBA API credentials in .env to load live standings.")


with tab2:
    st.subheader("Championship Odds (Monte Carlo · 10,000 simulations)")

    if data_loaded and not team_stats.empty:
        teams_list = team_stats["TEAM_NAME"].tolist() if "TEAM_NAME" in team_stats.columns else []

        if teams_list:
            win_probs = {}
            stats_map = {}
            if "TEAM_NAME" in team_stats.columns and "NET_RATING" in team_stats.columns:
                for _, row in team_stats.iterrows():
                    stats_map[row["TEAM_NAME"]] = row["NET_RATING"]

            for i, a in enumerate(teams_list):
                for b in teams_list[i+1:]:
                    rat_a = stats_map.get(a, 0)
                    rat_b = stats_map.get(b, 0)
                    diff = rat_a - rat_b
                    prob = 1 / (1 + np.exp(-diff * 0.15))
                    win_probs[(a, b)] = prob

            with st.spinner("Running 10,000 bracket simulations..."):
                sim_results = run_playoff_simulation(teams_list, win_probs)
                sim_df = results_to_dataframe(sim_results)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    sim_df.head(10),
                    x="championship_probability", y="team",
                    orientation="h",
                    color="championship_probability",
                    color_continuous_scale="Blues",
                    title="Top 10 Championship Probabilities",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.dataframe(
                    sim_df[["team", "championship_probability", "finals_probability"]],
                    use_container_width=True,
                )
    else:
        st.info("Connect NBA API credentials to run bracket simulation.")


with tab3:
    st.subheader("News Sentiment by Team (Credibility-Weighted)")

    if team_sentiment:
        sent_df = pd.DataFrame([
            {"team": k, "sentiment": v}
            for k, v in team_sentiment.items() if v != 0.0
        ]).sort_values("sentiment", ascending=False)

        if not sent_df.empty:
            fig = px.bar(
                sent_df,
                x="team", y="sentiment",
                color="sentiment",
                color_continuous_scale="RdYlGn",
                title="Media Sentiment per Team (positive = favourable coverage)",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Injury Alerts")
    if not scored_news.empty:
        injuries = flag_injury_alerts(scored_news)
        if not injuries.empty:
            st.dataframe(
                injuries[["title", "source", "credibility", "published_at"]].head(10),
                use_container_width=True,
            )
        else:
            st.success("No injury signals detected in recent coverage.")
    else:
        st.info("Connect NEWS_API_KEY and Reddit credentials in .env to load news.")


with tab4:
    st.subheader("Head-to-Head Matchup Predictor")

    if data_loaded and not team_stats.empty and "TEAM_NAME" in team_stats.columns:
        team_names = team_stats["TEAM_NAME"].tolist()
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("Home Team", team_names, index=0)
        with col2:
            away_options = [t for t in team_names if t != home_team]
            away_team = st.selectbox("Away Team", away_options, index=0)

        if st.button("Predict Matchup"):
            stats_map = {}
            for _, row in team_stats.iterrows():
                stats_map[row["TEAM_NAME"]] = row.get("NET_RATING", 0)

            rat_home = stats_map.get(home_team, 0)
            rat_away = stats_map.get(away_team, 0)
            diff = rat_home - rat_away
            sent_diff = team_sentiment.get(home_team.lower(), 0) - team_sentiment.get(away_team.lower(), 0)
            combined = diff * 0.8 + sent_diff * 0.2
            prob_home = round(1 / (1 + np.exp(-combined * 0.15)), 4)
            prob_away = round(1 - prob_home, 4)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{home_team} Win Probability", f"{prob_home:.1%}")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_home * 100,
                    title={"text": home_team},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "green"}},
                ))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.metric(f"{away_team} Win Probability", f"{prob_away:.1%}")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_away * 100,
                    title={"text": away_team},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "blue"}},
                ))
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Connect NBA API credentials to enable matchup predictor.")


st.sidebar.header("Configuration")
st.sidebar.markdown("""
Set the following in a `.env` file:
```
NEWS_API_KEY=your_key
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
```
NBA stats are fetched via the public `nba_api` — no key required.
""")
st.sidebar.divider()
st.sidebar.caption("Pipeline: Kalman Filter → XGBoost + RF Stacking → Monte Carlo")
