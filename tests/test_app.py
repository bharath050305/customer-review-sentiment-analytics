"""Smoke tests: every dashboard page renders without exceptions (Streamlit AppTest)."""
import pytest
from streamlit.testing.v1 import AppTest

from src.preprocess import ROOT

pytestmark = pytest.mark.skipif(not (ROOT / "models" / "sentiment_model.joblib").exists(), reason="model not trained yet")
APP = str(ROOT / "app.py")


@pytest.mark.parametrize("page", ["overview", "analyze", "batch", "performance", "insights", "about"])
def test_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["page"] = page
    if page == "analyze":
        at.session_state["review_text"] = "Stopped working after two weeks. Terrible, do not buy."
    if page == "batch":
        at.session_state["use_sample"] = True
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_analyze_predicts_negative():
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["page"] = "analyze"
    at.session_state["review_text"] = "Stopped working after two weeks. Terrible, do not buy."
    at.run()
    assert any("NEGATIVE" in m.value for m in at.markdown)


def test_analyze_empty_input_shows_hint():
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["page"] = "analyze"
    at.run()
    assert not at.exception
    assert any("Enter a review" in m.value for m in at.markdown)
