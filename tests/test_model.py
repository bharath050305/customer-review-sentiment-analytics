"""Sanity tests for the trained model artefacts (run `python -m src.train` first)."""
import json

import joblib
import numpy as np
import pytest

from src.preprocess import LABELS, ROOT

MODEL = ROOT / "models" / "sentiment_model.joblib"
METRICS = ROOT / "outputs" / "metrics.json"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="model not trained yet")


@pytest.fixture(scope="module")
def pipe():
    return joblib.load(MODEL)


@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS.read_text())


@pytest.mark.parametrize("text,expected", [
    ("Absolutely love this product, works perfectly and arrived early!", "positive"),
    ("Broke after two days. Total waste of money, do not buy.", "negative"),
    ("Terrible quality, very disappointed and I want a refund.", "negative"),
    ("Excellent value, highly recommend, five stars.", "positive"),
])
def test_obvious_reviews(pipe, text, expected):
    assert pipe.predict([text])[0] == expected


def test_negation_is_handled(pipe):
    pos = pipe.predict_proba(["This is good"])[0]
    neg = pipe.predict_proba(["This is not good"])[0]
    classes = list(pipe.named_steps["clf"].classes_)
    assert neg[classes.index("negative")] > pos[classes.index("negative")]


def test_probabilities_are_valid(pipe):
    proba = pipe.predict_proba(["It is okay", "great", "awful", ""])
    assert proba.shape == (4, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0).all()


def test_classes_match_labels(pipe):
    assert sorted(pipe.named_steps["clf"].classes_) == sorted(LABELS)


def test_handles_unseen_and_non_english_text(pipe):
    out = pipe.predict(["zzzz qqqq xxxx", "यह बहुत अच्छा है", "😀😀😀"])
    assert len(out) == 3 and set(out) <= set(LABELS)


def test_metrics_file_is_consistent(metrics):
    assert 0.65 <= metrics["test"]["accuracy"] <= 0.80
    assert metrics["test"]["macro_f1"] > metrics["model_comparison_val"][0]["macro_f1"]  # beats the majority baseline
    cm = np.array(metrics["confusion_matrix"]["matrix"])
    assert cm.sum() == metrics["dataset"]["test"]
    assert set(metrics["per_class"]) == set(LABELS)


def test_no_class_is_ignored(metrics):
    # macro-F1 would collapse if the model never predicted a class
    assert all(metrics["per_class"][c]["recall"] > 0.3 for c in LABELS)
