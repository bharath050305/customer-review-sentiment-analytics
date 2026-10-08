"""Shared data loading + text cleaning used by the training script, the notebook and the Streamlit app."""
import html
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "amazon_reviews_raw.jsonl"

LABELS = ["negative", "neutral", "positive"]
LABEL_COLORS = {"negative": "#EF4444", "neutral": "#F59E0B", "positive": "#10B981"}

# Negations and intensifiers carry sentiment ("not good", "very bad"), so they are NOT removed as stop-words.
KEEP_WORDS = {
    "no", "not", "nor", "never", "none", "nothing", "nobody", "neither", "cannot", "without",
    "very", "too", "more", "most", "less", "least", "much", "enough", "but", "however", "although",
}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS - KEEP_WORDS)

_URL = re.compile(r"https?://\S+|www\.\S+")
_TAG = re.compile(r"<[^>]+>")
_NON_ALPHA = re.compile(r"[^a-z\s']")
_SPACES = re.compile(r"\s+")
_CONTRACTIONS = [
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'(re|ve|ll|d|m|s)\b"), ""),
]


def clean_text(text: str) -> str:
    """Lower-case, strip HTML/URLs/punctuation/digits, expand negative contractions (don't -> do not)."""
    text = html.unescape(str(text)).lower()
    text = _URL.sub(" ", text)
    text = _TAG.sub(" ", text)
    for pattern, repl in _CONTRACTIONS:
        text = pattern.sub(repl, text)
    text = _NON_ALPHA.sub(" ", text).replace("'", "")
    return _SPACES.sub(" ", text).strip()


def stars_to_sentiment(stars: int) -> str:
    """1-2 stars -> negative, 3 -> neutral, 4-5 -> positive."""
    return "negative" if stars <= 2 else "neutral" if stars == 3 else "positive"


def load_reviews(path: Path = RAW_PATH, nrows: int | None = None) -> pd.DataFrame:
    """Load the raw JSONL, derive stars + 3-class sentiment, drop empty and duplicate reviews."""
    df = pd.read_json(path, lines=True, nrows=nrows)
    df = df.rename(columns={"text": "review"})
    df["stars"] = df["label"].astype(int) + 1
    df["sentiment"] = df["stars"].map(stars_to_sentiment)
    df["review"] = df["review"].astype(str).str.strip()
    df = df[df["review"].str.len() > 0].drop_duplicates(subset="review").reset_index(drop=True)
    df["word_count"] = df["review"].str.split().str.len()
    return df[["id", "review", "stars", "sentiment", "word_count"]]
