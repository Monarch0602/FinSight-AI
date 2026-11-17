
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


FEATURES = [
    "Daily_Return",
    "RSI",
    "MACD",
    "MACD_Hist",
    "EMA20_Gap",
    "EMA50_Gap",
    "Volatility_10",
    "Volume_Change"
]


def prepare_features(data):

    df = data.copy()

    # Daily returns
    df["Daily_Return"] = df["Close"].pct_change()

    # Technical indicator features
    df["EMA20_Gap"] = (
        df["Close"] / df["EMA20"] - 1
    )

    df["EMA50_Gap"] = (
        df["Close"] / df["EMA50"] - 1
    )

    # Short-term volatility
    df["Volatility_10"] = (
        df["Daily_Return"].rolling(10).std()
    )

    # Volume movement
    df["Volume_Change"] = (
        df["Volume"].pct_change()
    )

    # Target: next trading day's direction
    df["Target"] = (
        df["Close"].shift(-1) > df["Close"]
    ).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def train_market_model(data):

    df = prepare_features(data)

    # Remove incomplete observations
    df = df.dropna(subset=FEATURES)

    if len(df) < 100:
        raise ValueError(
            "Insufficient data. Select a longer analysis period."
        )

    # Exclude final row from training because
    # its next-day target is not yet observable.
    training_data = df.iloc[:-1].copy()

    X = training_data[FEATURES]
    y = training_data["Target"]

    if y.nunique() < 2:
        raise ValueError(
            "Historical dataset contains only one target class."
        )

    # Chronological split: no random shuffling
    split = int(len(training_data) * 0.80)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    if y_test.nunique() < 2:
        raise ValueError(
            "Test period needs examples from both classes."
        )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Train on historical training period
    model.fit(X_train, y_train)

    # Evaluate on unseen later observations
    predictions = model.predict(X_test)

    evaluation = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        )
    }

    # Refit on all available labeled historical observations
    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(X, y)

    # Predict using the latest available feature row
    latest_features = df[FEATURES].iloc[[-1]]

    probabilities = final_model.predict_proba(
        latest_features
    )[0]

    upward_probability = float(
        probabilities[list(final_model.classes_).index(1)]
    )

    # Feature importance
    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": final_model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    return {
        "evaluation": evaluation,
        "upward_probability": upward_probability,
        "feature_importance": importance,
        "model": final_model
    }
