"""Train, tune, evaluate and save the sentiment model.

Run from the project root:  python -m src.train
Outputs: models/sentiment_model.joblib, outputs/metrics.json, outputs/figures/*.png, data/sample_reviews.csv
"""
import json
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

from src.modeling import SEED, make_vectorizer, scores
from src.preprocess import LABEL_COLORS, LABELS, ROOT, STOP_WORDS, clean_text, load_reviews

warnings.filterwarnings("ignore")
OUT = ROOT / "outputs"
FIG = OUT / "figures"
for d in (OUT, FIG, ROOT / "models"):
    d.mkdir(parents=True, exist_ok=True)


def eda_stats(df):
    """Pre-computed EDA numbers so the dashboard works without shipping the 47 MB raw dataset."""
    edges = np.arange(0, 155, 5)
    hist = {lab: np.histogram(df.loc[df.sentiment == lab, "word_count"].clip(upper=150), bins=edges)[0].tolist()
            for lab in LABELS}
    top_bigrams = {}
    for lab in LABELS:
        texts = df.loc[df.sentiment == lab, "review"].sample(30_000, random_state=SEED)
        cv = CountVectorizer(preprocessor=clean_text, stop_words=STOP_WORDS, ngram_range=(2, 2), max_features=3000)
        counts = np.asarray(cv.fit_transform(texts).sum(axis=0)).ravel()
        order = np.argsort(counts)[::-1][:15]
        names = cv.get_feature_names_out()
        top_bigrams[lab] = [{"term": names[i], "count": int(counts[i])} for i in order]
    return {
        "star_counts": df["stars"].value_counts().sort_index().to_dict(),
        "length_hist": {"edges": edges.tolist(), "counts": hist},
        "length_stats": df.groupby("sentiment")["word_count"].agg(["mean", "median"]).round(1).to_dict("index"),
        "top_bigrams": top_bigrams,
    }


def main():
    t0 = time.time()
    df = load_reviews()
    print(f"Loaded {len(df):,} unique reviews")

    # 70 / 10 / 20 stratified split: train / validation (model selection) / test (final, touched once)
    X_tmp, X_test, y_tmp, y_test = train_test_split(df["review"], df["sentiment"], test_size=0.20,
                                                    stratify=df["sentiment"], random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.125, stratify=y_tmp,
                                                      random_state=SEED)
    print(f"train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")

    # ---- 1. Model comparison on validation set -------------------------------------------------
    vec = make_vectorizer()
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)
    print(f"TF-IDF matrix: {Xtr.shape}, density {Xtr.nnz / np.prod(Xtr.shape):.5f}  ({time.time() - t0:.0f}s)")

    candidates = {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.3),
        "Complement Naive Bayes": ComplementNB(alpha=0.3),
        "SGD (modified Huber)": SGDClassifier(loss="modified_huber", alpha=2e-6, class_weight="balanced",
                                              max_iter=30, random_state=SEED),
        "Linear SVM": LinearSVC(C=0.1, class_weight="balanced", random_state=SEED),
        "Logistic Regression": LogisticRegression(C=2, class_weight="balanced", max_iter=300),
    }
    comparison = []
    for name, model in candidates.items():
        s = time.time()
        model.fit(Xtr, y_train)
        fit_s = time.time() - s
        comparison.append({"model": name, **scores(y_val, model.predict(Xva)), "train_seconds": fit_s})
        print(f"  {name:28s} acc={comparison[-1]['accuracy']:.4f} macroF1={comparison[-1]['macro_f1']:.4f} ({fit_s:.1f}s)")

    # ---- 2. Hyper-parameter tuning (validation set) --------------------------------------------
    tuning = []
    for C in (0.5, 1, 2, 5):
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=300).fit(Xtr, y_train)
        tuning.append({"model": "Logistic Regression", "C": C, **scores(y_val, m.predict(Xva))})
    for C in (0.03, 0.1, 0.3, 1):
        m = LinearSVC(C=C, class_weight="balanced", random_state=SEED).fit(Xtr, y_train)
        tuning.append({"model": "Linear SVM", "C": C, **scores(y_val, m.predict(Xva))})
    tune_df = pd.DataFrame(tuning)
    print(tune_df.round(4).to_string(index=False))
    best_lr_C = float(tune_df[tune_df.model == "Logistic Regression"].sort_values("macro_f1").iloc[-1]["C"])
    best_svm = tune_df[tune_df.model == "Linear SVM"].sort_values("macro_f1").iloc[-1]
    best_lr = tune_df[tune_df.model == "Logistic Regression"].sort_values("macro_f1").iloc[-1]

    # Prefer Logistic Regression when it is within 0.3 pt of the SVM: it gives calibrated probabilities
    # and directly interpretable coefficients, both used by the Streamlit app.
    chosen = "Logistic Regression" if best_lr["macro_f1"] >= best_svm["macro_f1"] - 0.003 else "Linear SVM"
    print(f"Selected model: {chosen} (LR C={best_lr_C})")

    # ---- 3. Final model: refit on train+val, evaluate ONCE on the held-out test set ------------
    X_fit, y_fit = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
    vec = make_vectorizer()
    Xfit = vec.fit_transform(X_fit)
    Xte = vec.transform(X_test)
    clf = LogisticRegression(C=best_lr_C, class_weight="balanced", max_iter=500).fit(Xfit, y_fit)
    pipe = Pipeline([("tfidf", vec), ("clf", clf)])

    y_pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)
    classes = list(clf.classes_)
    test = scores(y_test, y_pred)
    auc = roc_auc_score(label_binarize(y_test, classes=classes), proba, average="macro", multi_class="ovr")
    Yb, roc = label_binarize(y_test, classes=classes), {}
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(Yb[:, i], proba[:, i])
        keep = np.linspace(0, len(fpr) - 1, 120).astype(int)
        roc[c] = {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist(),
                  "auc": float(roc_auc_score(Yb[:, i], proba[:, i]))}
    report = classification_report(y_test, y_pred, labels=LABELS, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    print(f"TEST  acc={test['accuracy']:.4f} macroF1={test['macro_f1']:.4f} AUC={auc:.4f}")
    print(classification_report(y_test, y_pred, labels=LABELS, digits=4))

    # ---- 4. Interpretability: most indicative words per class ----------------------------------
    vocab = np.array(vec.get_feature_names_out())
    top_words = {}
    for i, c in enumerate(classes):
        idx = np.argsort(clf.coef_[i])[::-1][:25]
        top_words[c] = [{"word": vocab[j], "weight": float(clf.coef_[i][j])} for j in idx]

    # ---- 5. Figures ----------------------------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(pd.DataFrame(cm, index=LABELS, columns=LABELS), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrix.png", dpi=150); plt.close(fig)

    comp_df = pd.DataFrame(comparison).sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(comp_df["model"], comp_df["macro_f1"], color="#1f77b4")
    for y, v in enumerate(comp_df["macro_f1"]):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center")
    ax.set(xlabel="Macro F1 (validation set)", title="Model comparison", xlim=(0, 1))
    fig.tight_layout(); fig.savefig(FIG / "model_comparison.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, c in zip(axes, classes):
        w = top_words[c][:15][::-1]
        ax.barh([x["word"] for x in w], [x["weight"] for x in w], color=LABEL_COLORS[c])
        ax.set_title(f"Top words -> {c}")
    fig.tight_layout(); fig.savefig(FIG / "top_words.png", dpi=150); plt.close(fig)

    # ---- 6. Persist ----------------------------------------------------------------------------
    joblib.dump(pipe, ROOT / "models" / "sentiment_model.joblib", compress=3)
    metrics = {
        "dataset": {"reviews": len(df), "train": len(X_fit), "test": len(X_test),
                    "class_counts": df["sentiment"].value_counts().to_dict(),
                    "vocabulary_size": int(len(vocab)), "mean_words": float(df["word_count"].mean())},
        "model_comparison_val": comparison,
        "tuning_val": tuning,
        "selected_model": chosen,
        "best_C": best_lr_C,
        "test": {**test, "roc_auc_macro_ovr": float(auc)},
        "per_class": {k: report[k] for k in LABELS},
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "top_words": top_words,
        "roc": roc,
        "eda": eda_stats(df),
        "total_seconds": time.time() - t0,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # demo file for the app's batch-upload tab
    demo = pd.DataFrame({"review": X_test.iloc[:200].values, "actual_sentiment": y_test.iloc[:200].values})
    demo.to_csv(ROOT / "data" / "sample_reviews.csv", index=False)
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
