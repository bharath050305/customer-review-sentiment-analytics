"""Feature extraction + scoring helpers shared by train.py and the notebook."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from src.preprocess import STOP_WORDS, clean_text

SEED = 42


def make_vectorizer(**overrides):
    """TF-IDF over cleaned unigrams+bigrams (sublinear tf, rare/very common terms pruned)."""
    params = dict(preprocessor=clean_text, stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=3,
                  max_df=0.9, max_features=300_000, sublinear_tf=True, dtype=np.float32)
    params.update(overrides)
    return TfidfVectorizer(**params)


def scores(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted")}
