"""Customer Review Sentiment Analytics - Streamlit dashboard.   Run:  streamlit run app.py"""
import html
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.preprocess import LABEL_COLORS, LABELS, clean_text  # clean_text is also needed to unpickle the model

ROOT = Path(__file__).parent
OUT, FIG = ROOT / "outputs", ROOT / "outputs" / "figures"
EMOJI = {"negative": "😞", "neutral": "😐", "positive": "😀"}
INK, MUTED, PRIMARY = "#0F172A", "#64748B", "#4F46E5"
PAGES = {
    "overview": "🏠  Overview",
    "analyze": "🔍  Analyze Review",
    "batch": "📂  Batch Analysis",
    "performance": "📊  Model Performance",
    "insights": "🗂️  Data Insights",
    "about": "ℹ️  About",
}
EXAMPLES = {
    "😀 Positive": "Absolutely love this blender! Powerful, easy to clean and it looks great on my counter. Worth every penny.",
    "😐 Neutral": "It works fine, nothing special. Does the job but the build quality is just average for the price.",
    "😞 Negative": "Stopped working after two weeks. Customer service never replied and I am not getting my money back. Terrible.",
    "🧠 Tricky": "I can not say I am happy with this. It is not as good as the description promised.",
}

st.set_page_config(page_title="Review Sentiment Analytics", page_icon="💬", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(ROOT / "models" / "sentiment_model.joblib")


@st.cache_data
def load_metrics():
    return json.loads((OUT / "metrics.json").read_text())


@st.cache_data
def _read_csv(path, mtime):  # mtime in the cache key -> refreshed when the notebook re-exports a file
    return pd.read_csv(path)


def load_csv(name):
    p = OUT / name
    return _read_csv(str(p), p.stat().st_mtime) if p.exists() else None


try:
    pipe, M = load_model(), load_metrics()
except FileNotFoundError:
    st.error("Model not found. Run `python -m src.train` first.")
    st.stop()
classes = list(pipe.named_steps["clf"].classes_)
T = M["test"]


# ----------------------------------------------------------------------------- helpers
def kpi(label, value, note="", accent=PRIMARY):
    return (f'<div class="kpi" style="--accent:{accent}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="note">{note}</div></div>')


def header(icon, title, sub):
    st.markdown(f'<div class="page-title"><span style="font-size:1.8rem">{icon}</span><h2>{title}</h2></div>'
                f'<p class="page-sub">{sub}</p>', unsafe_allow_html=True)


def style(fig, h=340, legend=True):
    fig.update_layout(height=h, margin=dict(l=12, r=16, t=44, b=78 if legend else 14), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color=INK, size=13), showlegend=legend,
                      legend=dict(orientation="h", y=-0.32, x=0.5, xanchor="center"))
    if fig.layout.title.text:
        fig.update_layout(title_font=dict(size=15, color=INK), title_x=0.02)
    fig.update_xaxes(gridcolor="#EEF2F7", zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False, automargin=True)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def prob_bars(proba):
    rows = ""
    for lab in LABELS:
        p = proba[classes.index(lab)]
        rows += (f'<div class="pbar"><div class="row"><span>{EMOJI[lab]} {lab.capitalize()}</span><span>{p:.1%}</span></div>'
                 f'<div class="track"><div class="fill" style="width:{p * 100:.1f}%;background:{LABEL_COLORS[lab]}"></div></div></div>')
    return rows


def explain(text):
    """Per-term evidence: tf-idf value x class coefficients, for unigrams (highlighting) and bigrams (drivers)."""
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    x = vec.transform([text])
    if x.nnz == 0:
        return {}
    names = np.array(vec.get_feature_names_out())[x.indices]
    coef = clf.coef_[:, x.indices] * x.data  # (classes, terms)
    return {n: coef[:, i] for i, n in enumerate(names)}


def highlight_html(text, ev):
    """Colour each word by the class it pushes towards; opacity = strength."""
    if not ev:
        return html.escape(text)
    strength = {t: (w.max() - np.delete(w, w.argmax()).mean(), classes[int(w.argmax())]) for t, w in ev.items() if " " not in t}
    top = max((s for s, _ in strength.values()), default=1e-9) or 1e-9
    out = []
    for tok in re.findall(r"\w+|\W+", text):
        key = clean_text(tok).strip()
        if tok.strip() and key in strength and strength[key][0] > 0.12 * top:
            s, cls = strength[key]
            a = 0.18 + 0.5 * min(1, s / top)
            c = LABEL_COLORS[cls].lstrip("#")
            rgb = tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
            out.append(f'<mark style="background:rgba({rgb[0]},{rgb[1]},{rgb[2]},{a:.2f})" title="pushes towards {cls}">{html.escape(tok)}</mark>')
        else:
            out.append(html.escape(tok))
    return "".join(out)


def meter(proba):
    score = (proba[classes.index("positive")] - proba[classes.index("negative")]) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(font=dict(size=34, color=INK), suffix=""),
        title=dict(text="Sentiment meter (−100 … +100)", font=dict(size=13, color=MUTED)),
        gauge=dict(axis=dict(range=[-100, 100], tickwidth=0, tickvals=[-100, -50, 0, 50, 100]),
                   bar=dict(color=INK, thickness=0.22), bgcolor="white", borderwidth=0,
                   steps=[dict(range=[-100, -25], color="#FCA5A5"), dict(range=[-25, 25], color="#FDE68A"),
                          dict(range=[25, 100], color="#6EE7B7")])))
    return style(fig, 230, legend=False)


def predict_df(texts):
    proba = pipe.predict_proba(texts)
    out = pd.DataFrame(proba, columns=classes)[LABELS]
    out.insert(0, "predicted_sentiment", [classes[i] for i in proba.argmax(axis=1)])
    out.insert(1, "confidence", proba.max(axis=1))
    return out


def go_to(page):
    st.session_state["page"] = page


def set_example(txt):
    st.session_state["review_text"] = txt


# ----------------------------------------------------------------------------- sidebar
st.session_state.setdefault("page", st.query_params.get("page", "overview"))
if "review_text" not in st.session_state and st.query_params.get("example") in [k.split(" ", 1)[1].lower() for k in EXAMPLES]:
    st.session_state["review_text"] = next(v for k, v in EXAMPLES.items() if k.split(" ", 1)[1].lower() == st.query_params["example"])
if st.session_state["page"] not in PAGES:
    st.session_state["page"] = "overview"

with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">💬</div><div><b>Review Sentiment</b><span>Analytics Studio</span></div></div>',
                unsafe_allow_html=True)
    st.radio("Navigate", list(PAGES), format_func=PAGES.get, key="page", label_visibility="collapsed")
    st.markdown(f'<div class="side-card"><div class="k">Test accuracy</div><div class="v">{T["accuracy"]:.1%}</div></div>'
                f'<div class="side-card"><div class="k">Macro F1</div><div class="v">{T["macro_f1"]:.3f}</div></div>'
                f'<div class="side-card"><div class="k">Trained on</div><div class="v">{M["dataset"]["train"]:,}</div>'
                f'<div class="k" style="margin-top:.1rem">reviews · {M["selected_model"]}</div></div>', unsafe_allow_html=True)
page = st.session_state["page"]


# ============================================================================= OVERVIEW
if page == "overview":
    st.markdown(f"""<div class="hero"><div class="bubble"></div>
      <h1>Customer Review<br>Sentiment Analytics</h1>
      <p>Turn {M['dataset']['reviews']:,} raw Amazon product reviews into actionable insight. A machine-learning pipeline reads each review and
      classifies it as <b>positive</b>, <b>neutral</b> or <b>negative</b> - and shows <i>why</i>.</p>
      <div class="chips"><span>🐍 Python</span><span>🐼 Pandas · NumPy</span><span>🤖 Scikit-learn</span><span>📓 Jupyter</span><span>🎈 Streamlit</span></div>
    </div>""", unsafe_allow_html=True)

    c = st.columns(4)
    c[0].markdown(kpi("Test accuracy", f"{T['accuracy']:.1%}", f"baseline {max(M['dataset']['class_counts'].values()) / M['dataset']['reviews']:.0%} (majority class)"), unsafe_allow_html=True)
    c[1].markdown(kpi("Macro F1-score", f"{T['macro_f1']:.3f}", "all classes weighted equally", "#7C3AED"), unsafe_allow_html=True)
    c[2].markdown(kpi("ROC-AUC", f"{T['roc_auc_macro_ovr']:.3f}", "one-vs-rest, macro average", "#0EA5E9"), unsafe_allow_html=True)
    c[3].markdown(kpi("Reviews analysed", f"{M['dataset']['reviews']:,}", f"{M['dataset']['vocabulary_size']:,} word features", "#10B981"), unsafe_allow_html=True)

    st.markdown("### How it works")
    steps = [("1", "Collect", "~200k Amazon reviews with 1-5 star ratings, mapped to 3 sentiment classes."),
             ("2", "Clean", "Strip HTML & URLs, expand <i>don't → do not</i>, keep negations."),
             ("3", "Vectorise", "TF-IDF on words + bigrams → sparse matrix (99.98 % zeros)."),
             ("4", "Classify", "Logistic Regression, tuned on a validation set, tested once."),
             ("5", "Explain", "Word-level highlights show what drove each prediction.")]
    for col, (n, t, d) in zip(st.columns(5), steps):
        col.markdown(f'<div class="step"><div class="n">{n}</div><b>{t}</b><span>{d}</span></div>', unsafe_allow_html=True)

    st.markdown("### Try it")
    cols = st.columns([1, 1, 1, 1, 1.2])
    for col, (name, txt) in zip(cols, EXAMPLES.items()):
        col.button(name, key=f"ov_{name}", width="stretch", on_click=lambda t=txt: (set_example(t), go_to("analyze")))
    cols[4].button("Open analyzer →", type="primary", width="stretch", on_click=go_to, args=("analyze",))

    left, right = st.columns([1.1, 1])
    with left:
        counts = pd.Series(M["dataset"]["class_counts"]).reindex(LABELS)
        fig = go.Figure(go.Pie(labels=[l.capitalize() for l in counts.index], values=counts.values, hole=0.62, sort=False,
                               marker=dict(colors=[LABEL_COLORS[l] for l in counts.index], line=dict(color="white", width=3)),
                               textinfo="percent", hovertemplate="%{label}: %{value:,}<extra></extra>"))
        fig.add_annotation(text=f"<b>{M['dataset']['reviews'] / 1000:.0f}k</b><br>reviews", showarrow=False, font=dict(size=16))
        show(style(fig.update_layout(title="Class distribution"), 330))
    with right:
        pc = pd.DataFrame(M["per_class"]).T.reindex(LABELS)
        fig = go.Figure(go.Bar(x=[l.capitalize() for l in pc.index], y=pc["f1-score"], marker_color=[LABEL_COLORS[l] for l in pc.index],
                               text=[f"{v:.2f}" for v in pc["f1-score"]], textposition="outside"))
        fig.update_yaxes(range=[0, 1])
        show(style(fig.update_layout(title="F1-score per class (test set)"), 330, legend=False))


# ============================================================================= ANALYZE
elif page == "analyze":
    header("🔍", "Analyze a review", "Paste any product review - the model predicts its sentiment and highlights the words that drove the decision.")
    cols = st.columns(len(EXAMPLES))
    for col, (name, txt) in zip(cols, EXAMPLES.items()):
        col.button(name, key=f"ex_{name}", width="stretch", on_click=set_example, args=(txt,))
    review = st.text_area("Review text", key="review_text", height=130, label_visibility="collapsed",
                          placeholder="Type or paste a customer review here... or pick an example above.")

    if not review.strip():
        st.markdown('<div class="card" style="text-align:center;color:#64748B;padding:2.2rem">'
                    '<div style="font-size:2.4rem">✍️</div>Enter a review or click an example to see the prediction.</div>', unsafe_allow_html=True)
    else:
        proba = pipe.predict_proba([review])[0]
        pred = classes[int(np.argmax(proba))]
        hist = st.session_state.setdefault("history", [])
        if not hist or hist[0]["review"] != review:
            hist.insert(0, {"review": review[:90] + ("…" if len(review) > 90 else ""), "sentiment": pred, "confidence": float(proba.max())})
            del hist[8:]

        left, right = st.columns([1.15, 1])
        with left:
            st.markdown(f'<div class="result" style="--c:{LABEL_COLORS[pred]}"><div class="emoji">{EMOJI[pred]}</div><div>'
                        f'<div class="lab">{pred.upper()}</div><div class="conf">Confidence <b>{proba.max():.1%}</b></div></div></div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="card" style="margin-top:1rem"><h4>Class probabilities</h4>{prob_bars(proba)}</div>', unsafe_allow_html=True)
        with right:
            show(meter(proba))

        ev = explain(review)
        st.markdown("### Why this prediction?")
        st.markdown(f'<div class="hl">{highlight_html(review, ev)}</div><div class="legend">Highlight colour = the sentiment a word pushes towards:'
                    f'<i style="background:{LABEL_COLORS["negative"]}"></i>negative<i style="background:{LABEL_COLORS["neutral"]}"></i>neutral'
                    f'<i style="background:{LABEL_COLORS["positive"]}"></i>positive · stronger colour = stronger evidence · '
                    f'negations (<em>not, never</em>) are kept and combine into bigrams like “not good”.</div>', unsafe_allow_html=True)
        if ev:
            k = classes.index(pred)
            df = pd.DataFrame({"term": list(ev), "contribution": [w[k] for w in ev.values()]}).sort_values("contribution")
            df = pd.concat([df.head(4), df.tail(8)]).drop_duplicates("term")
            fig = go.Figure(go.Bar(x=df["contribution"], y=df["term"], orientation="h",
                                   marker_color=[LABEL_COLORS[pred] if v > 0 else "#94A3B8" for v in df["contribution"]]))
            show(style(fig.update_layout(title=f"Terms supporting ({pred}) vs. opposing - contribution to the “{pred}” score"),
                       max(260, 30 * len(df) + 70), legend=False))

        if len(hist) > 1:
            st.markdown("### Recent analyses")
            h = pd.DataFrame(hist)
            st.dataframe(h.style.format({"confidence": "{:.1%}"}).map(
                lambda v: f"background-color:{LABEL_COLORS.get(v, '#fff')}22;font-weight:700;color:{LABEL_COLORS.get(v, INK)}", subset=["sentiment"]),
                width="stretch", hide_index=True)


# ============================================================================= BATCH
elif page == "batch":
    header("📂", "Batch analysis", "Upload a CSV of reviews and get sentiment for every row - instantly, with a downloadable report.")
    c1, c2 = st.columns([2, 1])
    up = c1.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
    c2.markdown('<div class="card"><h4>No file handy?</h4><div class="sub">Try 200 real held-out test reviews (includes the true labels so accuracy is shown).</div></div>',
                unsafe_allow_html=True)
    if c2.button("Use sample file", type="primary", width="stretch"):
        st.session_state["use_sample"] = True
    data = None
    if up is not None:
        data = pd.read_csv(up)
        st.session_state["use_sample"] = False
    elif st.session_state.get("use_sample"):
        data = pd.read_csv(ROOT / "data" / "sample_reviews.csv")

    if data is None:
        st.markdown('<div class="card" style="text-align:center;color:#64748B;padding:2.2rem"><div style="font-size:2.4rem">📄</div>'
                    'Upload a CSV with a column of review text to begin.</div>', unsafe_allow_html=True)
    else:
        text_cols = [c for c in data.columns if not pd.api.types.is_numeric_dtype(data[c])] or list(data.columns)
        col = st.selectbox("Column containing the review text", text_cols, index=text_cols.index("review") if "review" in text_cols else 0)
        res = pd.concat([data.reset_index(drop=True), predict_df(data[col].fillna("").astype(str))], axis=1)
        counts = res["predicted_sentiment"].value_counts().reindex(LABELS, fill_value=0)
        k = st.columns(4 if "actual_sentiment" not in res.columns else 5)
        k[0].markdown(kpi("Reviews", f"{len(res):,}"), unsafe_allow_html=True)
        for c, lab in zip(k[1:4], LABELS):
            c.markdown(kpi(f"{EMOJI[lab]} {lab.capitalize()}", f"{counts[lab]:,}", f"{counts[lab] / len(res):.0%} of reviews", LABEL_COLORS[lab]),
                       unsafe_allow_html=True)
        if "actual_sentiment" in res.columns:
            acc = (res["actual_sentiment"] == res["predicted_sentiment"]).mean()
            k[4].markdown(kpi("Accuracy", f"{acc:.1%}", "vs. true labels", "#7C3AED"), unsafe_allow_html=True)

        a, b = st.columns(2)
        with a:
            fig = go.Figure(go.Pie(labels=[l.capitalize() for l in counts.index], values=counts.values, hole=0.6, sort=False,
                                   marker=dict(colors=[LABEL_COLORS[l] for l in counts.index], line=dict(color="white", width=3))))
            show(style(fig.update_layout(title="Sentiment mix"), 300))
        with b:
            fig = go.Figure()
            for lab in LABELS:
                fig.add_trace(go.Histogram(x=res.loc[res.predicted_sentiment == lab, "confidence"], name=lab.capitalize(),
                                           marker_color=LABEL_COLORS[lab], xbins=dict(start=0.3, end=1.0, size=0.05), opacity=0.85))
            fig.update_layout(barmode="stack", xaxis_title="Model confidence")
            show(style(fig.update_layout(title="Confidence distribution"), 300))

        flt = st.multiselect("Filter by predicted sentiment", LABELS, default=LABELS)
        view = res[res.predicted_sentiment.isin(flt)].copy()
        view["confidence"] = view["confidence"].round(3)
        shown = view.head(500)
        st.caption(f"Showing {len(shown):,} of {len(view):,} rows" + (" (first 500; the download contains all rows)" if len(view) > 500 else ""))
        st.dataframe(shown.style.map(lambda v: f"background-color:{LABEL_COLORS.get(v, '#fff')}22;font-weight:700;color:{LABEL_COLORS.get(v, INK)}",
                                     subset=["predicted_sentiment"]),
                     width="stretch", height=360, hide_index=True,
                     column_config={"confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1, format="%.2f"),
                                    **{l: st.column_config.NumberColumn(l, format="%.2f") for l in LABELS}})
        st.download_button("⬇️  Download predictions (CSV)", res.to_csv(index=False).encode(), "sentiment_predictions.csv", "text/csv", type="primary")


# ============================================================================= PERFORMANCE
elif page == "performance":
    header("📊", "Model performance", f"Results on {M['dataset']['test']:,} held-out reviews that were never used for training, tuning or model selection.")
    c = st.columns(4)
    c[0].markdown(kpi("Accuracy", f"{T['accuracy']:.2%}"), unsafe_allow_html=True)
    c[1].markdown(kpi("Macro F1", f"{T['macro_f1']:.3f}", accent="#7C3AED"), unsafe_allow_html=True)
    c[2].markdown(kpi("Weighted F1", f"{T['weighted_f1']:.3f}", accent="#0EA5E9"), unsafe_allow_html=True)
    c[3].markdown(kpi("ROC-AUC", f"{T['roc_auc_macro_ovr']:.3f}", accent="#10B981"), unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["🎯 Evaluation", "🏁 Model comparison", "📈 Scalability", "🧪 Experiments"])
    with t1:
        a, b = st.columns(2)
        with a:
            norm = st.toggle("Show as % of actual class (recall)", value=True)
            cm = np.array(M["confusion_matrix"]["matrix"], dtype=float)
            z = cm / cm.sum(1, keepdims=True) if norm else cm
            lab = [l.capitalize() for l in LABELS]
            fig = go.Figure(go.Heatmap(z=z, x=lab, y=lab, colorscale=[[0, "#EEF2FF"], [1, "#4338CA"]], showscale=False,
                                       text=[[f"{v:.0%}" if norm else f"{v:,.0f}" for v in r] for r in z], texttemplate="%{text}",
                                       textfont=dict(size=16)))
            fig.update_yaxes(autorange="reversed", title="Actual")
            fig.update_xaxes(title="Predicted")
            show(style(fig.update_layout(title="Confusion matrix"), 360, legend=False))
        with b:
            pc = pd.DataFrame(M["per_class"]).T.reindex(LABELS)
            fig = go.Figure()
            for met, col in [("precision", "#6366F1"), ("recall", "#A78BFA"), ("f1-score", "#F472B6")]:
                fig.add_trace(go.Bar(x=lab, y=pc[met], name=met.capitalize(), marker_color=col, text=[f"{v:.2f}" for v in pc[met]], textposition="outside"))
            fig.update_yaxes(range=[0, 1.05])
            show(style(fig.update_layout(title="Precision · Recall · F1 per class", barmode="group"), 360))
        st.markdown('<div class="card"><h4>Reading the results</h4><div class="sub">The model separates positive from negative well. '
                    '<b>Neutral (3★)</b> is the hardest class: such reviews mix praise and complaints, so they sit between the other two - '
                    'most remaining errors involve it. This is a property of star-derived labels, not just of the model.</div></div>', unsafe_allow_html=True)
        if M.get("roc"):
            fig = go.Figure()
            for cls, r in M["roc"].items():
                fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=f"{cls.capitalize()} (AUC {r['auc']:.3f})",
                                         line=dict(color=LABEL_COLORS[cls], width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#94A3B8"), name="Chance", showlegend=False))
            fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
            show(style(fig.update_layout(title="ROC curves (one-vs-rest)"), 380))
    with t2:
        cmp_df = pd.DataFrame(M["model_comparison_val"]).sort_values("macro_f1")
        a, b = st.columns([1.4, 1])
        with a:
            fig = go.Figure()
            fig.add_trace(go.Bar(y=cmp_df["model"], x=cmp_df["accuracy"], orientation="h", name="Accuracy", marker_color="#A5B4FC"))
            fig.add_trace(go.Bar(y=cmp_df["model"], x=cmp_df["macro_f1"], orientation="h", name="Macro F1", marker_color="#4F46E5",
                                 text=[f"{v:.3f}" for v in cmp_df["macro_f1"]], textposition="outside"))
            fig.update_xaxes(range=[0, 1])
            show(style(fig.update_layout(title="Validation-set comparison", barmode="group"), 420))
        with b:
            fig = go.Figure(go.Bar(y=cmp_df["model"], x=cmp_df["train_seconds"], orientation="h", marker_color="#F59E0B",
                                   text=[f"{v:.1f}s" for v in cmp_df["train_seconds"]], textposition="outside"))
            fig.update_xaxes(range=[0, cmp_df["train_seconds"].max() * 1.3])
            show(style(fig.update_layout(title="Training time (s)"), 420, legend=False))
        tune = pd.DataFrame(M["tuning_val"])
        fig = go.Figure()
        for mdl, col in [("Logistic Regression", "#4F46E5"), ("Linear SVM", "#EC4899")]:
            d = tune[tune.model == mdl]
            fig.add_trace(go.Scatter(x=d["C"], y=d["macro_f1"], mode="lines+markers", name=mdl, line=dict(color=col, width=3), marker=dict(size=9)))
        fig.update_layout(xaxis_title="Regularisation strength C (log scale)", yaxis_title="Validation macro-F1")
        fig.update_xaxes(type="log")
        show(style(fig.update_layout(title="Hyper-parameter tuning"), 320))
        st.caption(f"Selected: **{M['selected_model']}** (C = {M['best_C']}) - best tuned validation macro-F1 (Logistic Regression {tune[tune.model == 'Logistic Regression'].macro_f1.max():.3f} vs Linear SVM {tune[tune.model == 'Linear SVM'].macro_f1.max():.3f}); it also provides calibrated probabilities and interpretable coefficients.")
    with t3:
        sc = load_csv("scalability.csv")
        if sc is None:
            st.info("Run the notebook to generate the scalability study.")
        else:
            a, b = st.columns(2)
            with a:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sc.train_reviews, y=sc.macro_f1, mode="lines+markers", name="Macro F1", line=dict(color="#4F46E5", width=3), marker=dict(size=9)))
                fig.add_trace(go.Scatter(x=sc.train_reviews, y=sc.accuracy, mode="lines+markers", name="Accuracy", line=dict(color="#10B981", width=3), marker=dict(size=9)))
                fig.update_xaxes(type="log", title="Training reviews", tickvals=sc.train_reviews, ticktext=[f"{v / 1000:.0f}k" for v in sc.train_reviews])
                show(style(fig.update_layout(title="More data → better model"), 340))
            with b:
                fig = go.Figure(go.Scatter(x=sc.train_reviews, y=sc.vectorize_s + sc.fit_s, mode="lines+markers", line=dict(color="#F59E0B", width=3), marker=dict(size=9)))
                fig.update_xaxes(type="log", title="Training reviews", tickvals=sc.train_reviews, ticktext=[f"{v / 1000:.0f}k" for v in sc.train_reviews])
                fig.update_yaxes(type="log", title="Seconds", tickvals=[1, 2, 5, 10, 20, 50], ticktext=["1", "2", "5", "10", "20", "50"])
                show(style(fig.update_layout(title="Training time grows ~linearly"), 340, legend=False))
        st_json = OUT / "streaming.json"
        if st_json.exists():
            s = json.loads(st_json.read_text())
            st.markdown("#### Out-of-core streaming training")
            st.caption("The raw file is sorted by star rating, so a naive single pass forgets the early classes. An out-of-core shuffle-shard fixes it.")
            k = st.columns(4)
            k[0].markdown(kpi("Streamed", f"{s['reviews_streamed']:,}", f"chunks of {s['chunk_size']:,}", "#0EA5E9"), unsafe_allow_html=True)
            if "naive_accuracy" in s:
                k[1].markdown(kpi("Naive (sorted file)", f"{s['naive_accuracy']:.1%}", f"macro-F1 {s['naive_macro_f1']:.2f} - collapses", "#EF4444"), unsafe_allow_html=True)
            k[2].markdown(kpi("Shuffle-shard stream", f"{s['stream_accuracy']:.1%}", f"macro-F1 {s['stream_macro_f1']:.3f}", "#10B981"), unsafe_allow_html=True)
            k[3].markdown(kpi("In-memory model", f"{s['inmem_accuracy']:.1%}", f"macro-F1 {s['inmem_macro_f1']:.3f}", "#7C3AED"), unsafe_allow_html=True)
    with t4:
        ab = load_csv("ablation.csv")
        if ab is not None:
            st.markdown("#### Ablation: which design choices matter? (validation, 60k-review subsample)")
            fig = go.Figure(go.Bar(y=ab["setting"], x=ab["macro_f1"], orientation="h", marker_color=["#4F46E5" if "chosen" in s else "#C7D2FE" for s in ab["setting"]],
                                   text=[f"{v:.3f}" for v in ab["macro_f1"]], textposition="outside"))
            fig.update_xaxes(range=[ab["macro_f1"].min() - 0.03, ab["macro_f1"].max() + 0.02])
            fig.update_yaxes(autorange="reversed")
            show(style(fig, 330, legend=False))
        err = load_csv("error_examples.csv")
        if err is not None:
            st.markdown("#### Most confident mistakes (error analysis)")
            st.caption("Often label noise (a glowing review with 1★), sarcasm, or mixed feelings - limits of star-derived labels.")
            st.dataframe(err.head(12), width="stretch", hide_index=True,
                         column_config={"confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1, format="%.2f")})
        st.image(str(FIG / "top_words.png"), caption="Most indicative words per class (model coefficients)", width="stretch")


# ============================================================================= INSIGHTS
elif page == "insights":
    header("🗂️", "Dataset insights", "Exploratory analysis of the review corpus - what customers write and what they complain about.")
    D, E = M["dataset"], M.get("eda", {})
    k = st.columns(4)
    k[0].markdown(kpi("Unique reviews", f"{D['reviews']:,}"), unsafe_allow_html=True)
    k[1].markdown(kpi("Vocabulary", f"{D['vocabulary_size']:,}", "words + bigrams", "#7C3AED"), unsafe_allow_html=True)
    k[2].markdown(kpi("Avg. length", f"{D['mean_words']:.0f} words", "", "#0EA5E9"), unsafe_allow_html=True)
    k[3].markdown(kpi("Classes", "3", "negative · neutral · positive", "#10B981"), unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        if E:
            sc = pd.Series(E["star_counts"])
            cols = ["#EF4444", "#F97316", "#F59E0B", "#84CC16", "#10B981"]
            fig = go.Figure(go.Bar(x=[f"{int(i)}★" for i in sc.index], y=sc.values, marker_color=cols, text=[f"{v:,}" for v in sc.values], textposition="outside"))
            show(style(fig.update_layout(title="Star ratings (balanced at the source)"), 330, legend=False))
    with b:
        if E:
            h = E["length_hist"]
            mid = [(h["edges"][i] + h["edges"][i + 1]) / 2 for i in range(len(h["edges"]) - 1)]
            fig = go.Figure()
            for lab in LABELS:
                cnt = np.array(h["counts"][lab], dtype=float)
                fig.add_trace(go.Scatter(x=mid, y=cnt / cnt.sum(), mode="lines", name=lab.capitalize(), fill="tozeroy", line=dict(color=LABEL_COLORS[lab], width=2.5)))
            fig.update_layout(xaxis_title="Words per review (capped at 150)", yaxis_title="Share of reviews")
            show(style(fig.update_layout(title="Review length is similar across classes"), 330))
    if E:
        st.markdown("### Most frequent phrases per sentiment")
        for col, lab in zip(st.columns(3), LABELS):
            d = pd.DataFrame(E["top_bigrams"][lab]).head(10)[::-1]
            fig = go.Figure(go.Bar(x=d["count"], y=d["term"], orientation="h", marker_color=LABEL_COLORS[lab]))
            col.plotly_chart(style(fig.update_layout(title=lab.capitalize()), 380, legend=False), width="stretch", config={"displayModeBar": False})
    th = load_csv("themes.csv")
    if th is not None:
        st.markdown("### What do unhappy customers talk about?")
        th = th.sort_values("pct_negative")
        base = D["class_counts"]["negative"] / D["reviews"]
        fig = go.Figure(go.Bar(y=th["theme"], x=th["pct_negative"], orientation="h", marker_color="#EF4444", text=[f"{v:.0%}" for v in th["pct_negative"]], textposition="outside"))
        fig.add_vline(x=base, line_dash="dash", line_color=INK, annotation_text=f"overall {base:.0%}", annotation_position="top")
        fig.update_xaxes(tickformat=".0%", range=[0, max(th["pct_negative"]) * 1.2])
        show(style(fig.update_layout(title="Share of NEGATIVE reviews among reviews mentioning each theme"), 400, legend=False))
        st.caption("Themes above the dashed line are disproportionately linked to unhappy customers → priority areas for the business.")
    if (FIG / "eda_wordclouds.png").exists():
        st.markdown("### Word clouds")
        st.image(str(FIG / "eda_wordclouds.png"), width="stretch")


# ============================================================================= ABOUT
else:
    header("ℹ️", "About this project", "Big Data Analytics using Python and Machine Learning - Customer Review Sentiment Analytics.")
    a, b = st.columns([1.3, 1])
    with a:
        st.markdown(f"""<div class="card"><h4>Problem</h4><div class="sub">Online stores receive thousands of reviews daily; reading them by hand is impossible.
        This project classifies each review as <b>positive</b>, <b>neutral</b> or <b>negative</b> and surfaces what customers care about.</div>
        <h4 style="margin-top:1rem">Methodology</h4>
        <ol style="color:#334155;line-height:1.75;margin:0;padding-left:1.1rem">
        <li><b>Data</b> - {M['dataset']['reviews']:,} unique Amazon reviews; 1-2★ → negative, 3★ → neutral, 4-5★ → positive.</li>
        <li><b>Cleaning</b> - HTML/URL removal, lower-casing, contraction expansion; negations and intensifiers are <b>kept</b>.</li>
        <li><b>Features</b> - TF-IDF over unigrams + bigrams with sublinear term frequency.</li>
        <li><b>Models</b> - baseline, Multinomial/Complement NB, SGD, Linear SVM, Logistic Regression.</li>
        <li><b>Validation</b> - stratified 70/10/20 split; tune on validation, test set used once.</li>
        <li><b>Imbalance</b> - balanced class weights, macro-F1 as headline metric.</li>
        <li><b>Big-data techniques</b> - sparse matrices, scalability study, out-of-core streaming with hashing + <code>partial_fit</code>.</li>
        <li><b>Explainability</b> - coefficient × TF-IDF value → per-word evidence.</li></ol></div>""", unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><h4>Tech stack</h4><div style="margin-top:.5rem">'
                    + "".join(f'<span class="tag">{t}</span>' for t in ["Python 3", "Pandas", "NumPy", "Scikit-learn", "SciPy (sparse)", "Jupyter Notebook",
                                                                       "Streamlit", "Plotly", "Matplotlib", "Seaborn", "WordCloud", "Joblib"])
                    + '</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="card" style="margin-top:1rem"><h4>Known limitations</h4><div class="sub">'
                    '• Star ratings are noisy proxies for sentiment.<br>• Bag-of-words ignores word order, sarcasm and context.<br>'
                    '• Neutral (3★) reviews are inherently ambiguous.<br>• Next steps: transformers (DistilBERT), aspect-based sentiment, Spark/Dask at scale.</div></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="card" style="margin-top:1rem"><h4>Data source</h4><div class="sub">Amazon product reviews (English) - Hugging Face '
                    '<code>SetFit/amazon_reviews_multi_en</code>, used for educational purposes.</div></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Customer Review Sentiment Analytics · Big Data Analytics project · Python · Scikit-learn · Streamlit</div>', unsafe_allow_html=True)
