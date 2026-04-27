import os
import requests
import praw
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict


NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = "nba-playoff-predictor/1.0"

# Sources weighted by historical accuracy and proximity to the sport.
# Score 1.0 = most credible (beat reporters), 0.3 = low signal (fan blogs).
SOURCE_CREDIBILITY: Dict[str, float] = {
    "theathletic.com": 1.0,
    "espn.com": 0.9,
    "nba.com": 1.0,
    "bleacherreport.com": 0.7,
    "cbssports.com": 0.75,
    "SI.com": 0.75,
    "yahoo.com": 0.65,
    "reddit.com": 0.4,
}

# Verified beat reporters / analysts on Reddit (u/username)
TRUSTED_REDDIT_AUTHORS = {
    "wojespn", "ShamsCharania", "Adrian_Wojnarowski",
    "ChrisBHaynes", "MarcJSpears", "ramona_shelburne",
}


def fetch_news_articles(query: str = "NBA playoffs", days_back: int = 7) -> pd.DataFrame:
    if not NEWS_API_KEY:
        print("NEWS_API_KEY not set — skipping NewsAPI fetch.")
        return pd.DataFrame()

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = (
        f"https://newsapi.org/v2/everything"
        f"?q={query}&from={from_date}&sortBy=publishedAt"
        f"&language=en&apiKey={NEWS_API_KEY}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])

    rows = []
    for a in articles:
        source = a.get("source", {}).get("name", "").lower()
        credibility = _match_credibility(source)
        rows.append({
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "content": a.get("content", ""),
            "source": source,
            "credibility": credibility,
            "published_at": a.get("publishedAt", ""),
            "url": a.get("url", ""),
        })

    return pd.DataFrame(rows)


def fetch_reddit_posts(subreddits: List[str] = None, limit: int = 100) -> pd.DataFrame:
    if not REDDIT_CLIENT_ID:
        print("REDDIT_CLIENT_ID not set — skipping Reddit fetch.")
        return pd.DataFrame()

    if subreddits is None:
        subreddits = ["nba", "nbadiscussion", "nbaanalysis"]

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

    rows = []
    for sub in subreddits:
        for post in reddit.subreddit(sub).hot(limit=limit):
            author = str(post.author).lower() if post.author else ""
            credibility = 0.8 if author in TRUSTED_REDDIT_AUTHORS else 0.35
            rows.append({
                "title": post.title,
                "content": post.selftext,
                "source": f"reddit/r/{sub}",
                "author": author,
                "credibility": credibility,
                "score": post.score,
                "published_at": datetime.utcfromtimestamp(post.created_utc).isoformat(),
                "url": f"https://reddit.com{post.permalink}",
            })

    return pd.DataFrame(rows)


def _match_credibility(source_name: str) -> float:
    for domain, score in SOURCE_CREDIBILITY.items():
        if domain in source_name:
            return score
    return 0.3


def load_all_news(query: str = "NBA playoffs") -> pd.DataFrame:
    news = fetch_news_articles(query=query)
    reddit = fetch_reddit_posts()
    combined = pd.concat([news, reddit], ignore_index=True)
    combined = combined[combined["credibility"] >= 0.35]
    return combined.sort_values("credibility", ascending=False).reset_index(drop=True)
