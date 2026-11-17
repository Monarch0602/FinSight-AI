
import streamlit as st
from transformers import pipeline


@st.cache_resource
def load_finbert():
    model = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=-1
    )
    return model


def analyze_sentiment(headlines):

    if not headlines:
        return []

    model = load_finbert()

    predictions = model(
        headlines,
        truncation=True,
        max_length=256,
        top_k=None
    )

    results = []

    for headline, prediction in zip(headlines, predictions):

        scores = {
            item["label"].lower(): item["score"]
            for item in prediction
        }

        label = max(scores, key=scores.get)

        results.append({
            "headline": headline,
            "sentiment": label.capitalize(),
            "positive": scores.get("positive", 0),
            "negative": scores.get("negative", 0),
            "neutral": scores.get("neutral", 0),
            "confidence": scores[label]
        })

    return results
