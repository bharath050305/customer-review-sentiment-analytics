"""Unit tests for data loading and text cleaning."""
import pandas as pd
import pytest

from src.preprocess import KEEP_WORDS, LABELS, STOP_WORDS, clean_text, load_reviews, stars_to_sentiment


@pytest.mark.parametrize("stars,expected", [(1, "negative"), (2, "negative"), (3, "neutral"), (4, "positive"), (5, "positive")])
def test_star_mapping(stars, expected):
    assert stars_to_sentiment(stars) == expected


def test_labels_are_three_classes():
    assert LABELS == ["negative", "neutral", "positive"]


def test_clean_text_lowercases_and_strips_noise():
    out = clean_text("GREAT <b>product</b>!!! See http://spam.example.com 100%")
    assert out == "great product see"


def test_clean_text_expands_negative_contractions():
    assert clean_text("I don't like it, can't recommend, won't buy") == "i do not like it can not recommend will not buy"


def test_clean_text_handles_html_entities_and_empty():
    assert clean_text("Tom &amp; Jerry") == "tom jerry"
    assert clean_text("") == ""
    assert clean_text("12345 !!!") == ""


def test_negations_and_intensifiers_are_kept_out_of_stopwords():
    for w in ("not", "no", "never", "very", "too", "nothing"):
        assert w not in STOP_WORDS and w in KEEP_WORDS


def test_common_stopwords_are_removed():
    assert "the" in STOP_WORDS and "and" in STOP_WORDS


def test_load_reviews_dedupes_and_labels(tmp_path):
    p = tmp_path / "mini.jsonl"
    pd.DataFrame({"id": ["a", "b", "c", "d"], "text": ["Great!", "Great!", "  ", "Terrible"],
                  "label": [4, 4, 2, 0], "label_text": ["4", "4", "2", "0"]}).to_json(p, orient="records", lines=True)
    df = load_reviews(p)
    assert len(df) == 2  # duplicate and blank review dropped
    assert set(df["sentiment"]) == {"positive", "negative"}
    assert set(df["stars"]) == {5, 1}
    assert {"review", "stars", "sentiment", "word_count"} <= set(df.columns)
