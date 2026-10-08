"""Measure extra facts about the trained model for the report: latency, throughput, memory, error breakdown.

Run:  python -m src.facts   ->  outputs/facts.json
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.modeling import SEED
from src.preprocess import LABELS, ROOT, clean_text, load_reviews

EXAMPLE = "I DON'T like it!! Can't recommend, the seller never replied."


def main():
    df = load_reviews()
    X_tmp, X_test, y_tmp, y_test = train_test_split(df["review"], df["sentiment"], test_size=0.2, stratify=df["sentiment"], random_state=SEED)
    pipe = joblib.load(ROOT / "models" / "sentiment_model.joblib")
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]

    # ---- batch throughput + predictions on the full test set
    t = time.perf_counter(); proba = pipe.predict_proba(X_test); batch_s = time.perf_counter() - t
    pred = np.array(clf.classes_)[proba.argmax(1)]
    correct = pred == y_test.values

    # ---- single-review latency
    sample = X_test.sample(300, random_state=1).tolist()
    pipe.predict_proba(sample[:5])
    lat = []
    for s in sample:
        t = time.perf_counter(); pipe.predict_proba([s]); lat.append((time.perf_counter() - t) * 1000)

    # ---- error breakdown
    err = pd.DataFrame({"actual": y_test.values, "pred": pred, "correct": correct, "words": X_test.str.split().str.len().values})
    wrong = err[~err.correct]
    bucket = pd.cut(err.words, [0, 10, 25, 50, 100, 10_000], labels=["<=10", "11-25", "26-50", "51-100", ">100"])
    by_len = err.groupby(bucket, observed=True)["correct"].agg(["mean", "size"])
    pairs = wrong.groupby(["actual", "pred"]).size().sort_values(ascending=False)

    # ---- sparse-matrix memory
    Xte = vec.transform(X_test)
    sparse_mb = (Xte.data.nbytes + Xte.indices.nbytes + Xte.indptr.nbytes) / 1e6
    dense_gb = Xte.shape[0] * Xte.shape[1] * 4 / 1e9

    # ---- worked example
    xe = vec.transform([EXAMPLE])
    names = np.array(vec.get_feature_names_out())[xe.indices]
    pe = clf.predict_proba(xe)[0]
    top = clf.classes_[pe.argmax()]
    contrib = xe.data * clf.coef_[list(clf.classes_).index(top)][xe.indices]
    order = np.argsort(contrib)[::-1][:6]

    facts = {
        "raw_rows": int(sum(1 for _ in open(ROOT / "data" / "amazon_reviews_raw.jsonl", encoding="utf-8"))),
        "unique_rows": len(df),
        "batch": {"reviews": len(X_test), "seconds": batch_s, "reviews_per_second": len(X_test) / batch_s},
        "latency_ms": {"mean": float(np.mean(lat)), "median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)), "n": len(lat)},
        "errors": {"total": int(len(wrong)), "neutral_involved_share": float(((wrong.actual == "neutral") | (wrong.pred == "neutral")).mean()),
                   "pairs": {f"{a}->{b}": int(n) for (a, b), n in pairs.items()}},
        "accuracy_by_length": {str(k): {"accuracy": float(v["mean"]), "n": int(v["size"])} for k, v in by_len.iterrows()},
        "test_matrix": {"rows": Xte.shape[0], "features": Xte.shape[1], "nnz": int(Xte.nnz), "sparsity": 1 - Xte.nnz / np.prod(Xte.shape),
                        "sparse_mb": sparse_mb, "dense_gb": dense_gb, "avg_nnz_per_review": Xte.nnz / Xte.shape[0]},
        "model_file_mb": (ROOT / "models" / "sentiment_model.joblib").stat().st_size / 1e6,
        "example": {"text": EXAMPLE, "cleaned": clean_text(EXAMPLE), "predicted": top,
                    "proba": {c: float(p) for c, p in zip(clf.classes_, pe)},
                    "top_terms": [{"term": names[i], "contribution": float(contrib[i])} for i in order]},
    }
    (ROOT / "outputs" / "facts.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps({k: facts[k] for k in ("batch", "latency_ms", "errors", "test_matrix", "model_file_mb")}, indent=1))
    print(facts["example"])


if __name__ == "__main__":
    main()
