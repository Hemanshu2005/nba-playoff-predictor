import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier
import joblib
from typing import Tuple, Optional


FEATURES = [
    # Kalman-smoothed team metrics
    "NET_RATING_KF", "OFF_RATING_KF", "DEF_RATING_KF",
    "PACE_KF", "TS_PCT_KF", "PIE_KF", "W_PCT_KF",
    # Raw contextual features
    "REST_DAYS", "HOME_AWAY",
    # NLP sentiment score (credibility-weighted)
    "SENTIMENT_SCORE",
    # Head-to-head differentials (home - away)
    "NET_RATING_DIFF", "OFF_RATING_DIFF", "DEF_RATING_DIFF",
    "REST_DAYS_DIFF",
]

TARGET = "HOME_WIN"


def build_stacking_classifier(random_state: int = 42) -> StackingClassifier:
    """
    Two-level stacking classifier:
      Level 0 base learners:
        - XGBoost: gradient boosting, sequential error correction
        - Random Forest: bagging, parallel variance reduction
      Level 1 meta-learner:
        - Logistic Regression: interpretable, combines base predictions
    """
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )

    meta = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=random_state,
    )

    stacker = StackingClassifier(
        estimators=[("xgb", xgb), ("rf", rf)],
        final_estimator=meta,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )

    return stacker


def build_pipeline(random_state: int = 42) -> Pipeline:
    scaler = StandardScaler()
    stacker = build_stacking_classifier(random_state)
    return Pipeline([("scaler", scaler), ("model", stacker)])


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    available = [f for f in FEATURES if f in df.columns]
    X = df[available].copy()
    y = df[TARGET] if TARGET in df.columns else None
    X = X.fillna(X.median())
    return X, y


def train(df: pd.DataFrame, random_state: int = 42) -> Pipeline:
    X, y = prepare_features(df)
    pipeline = build_pipeline(random_state)
    pipeline.fit(X, y)
    return pipeline


def evaluate(pipeline: Pipeline, df: pd.DataFrame) -> dict:
    X, y = prepare_features(df)
    preds = pipeline.predict(X)
    proba = pipeline.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, preds),
        "log_loss": log_loss(y, proba),
        "roc_auc": roc_auc_score(y, proba),
    }


def predict_matchup(
    pipeline: Pipeline,
    home_features: dict,
    away_features: dict,
    sentiment_score: float = 0.0,
) -> dict:
    row = {}
    for k, v in home_features.items():
        row[k] = v
    for k, v in away_features.items():
        row[f"{k}_DIFF"] = home_features.get(k, 0) - v

    row["SENTIMENT_SCORE"] = sentiment_score
    row["HOME_AWAY"] = 1

    X = pd.DataFrame([row])
    available = [f for f in FEATURES if f in X.columns]
    X = X[available].fillna(0)

    win_prob = pipeline.predict_proba(X)[0][1]
    return {
        "home_win_probability": round(win_prob, 4),
        "away_win_probability": round(1 - win_prob, 4),
    }


def save_model(pipeline: Pipeline, path: str = "models/stacking_model.pkl") -> None:
    joblib.dump(pipeline, path)


def load_model(path: str = "models/stacking_model.pkl") -> Pipeline:
    return joblib.load(path)
