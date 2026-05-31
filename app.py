"""
app.py — "That Never Happened" Streamlit Interface
====================================================
Run with:
    streamlit run app.py

Requires gaslighting_detector.py (and gaslighting_data.py) in the same directory.
"""

import io
import json
import os
import contextlib
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="That Never Happened",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Imports from project
try:
    from gaslighting_detector import (
        MODEL_DIR,
        ACCUM_LABELS_PATH,
        REDDIT_CSV_PATH,
        load_or_train,
        models_exist,
        predict_conversations,
        retrain,
        train,
    )
    from gaslighting_data import append_labeled_data
    PROJECT_AVAILABLE = True
    _IMPORT_ERROR = ""
except ImportError as _err:
    PROJECT_AVAILABLE = False
    _IMPORT_ERROR = str(_err)
    # Fallback constants so the sidebar doesn't crash
    MODEL_DIR        = "saved_models"
    ACCUM_LABELS_PATH = "accumulated_labels.csv"



st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700'    '&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

_CSS = """
html, body, [data-testid="stAppViewContainer"] {
    background: #080C14 !important;
    color: #E2E8F0;
    font-family: 'DM Mono', monospace;
}
[data-testid="stSidebar"] {
    background: #0D1220 !important;
    border-right: 1px solid #1E2A40;
}
[data-testid="stSidebar"] * { color: #94A3B8; }
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    color: #F8FAFC !important;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 1.25rem;
}
.metric-box {
    background: #0D1525;
    border: 1px solid #1E2A40;
    border-radius: 10px;
    padding: 14px 16px;
}
.metric-box .mlabel {
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 4px;
}
.metric-box .mvalue {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1;
}
.mvalue.gas  { color: #F59E0B; }
.mvalue.safe { color: #10B981; }
.mvalue.hc   { color: #818CF8; }
.gas-card {
    background: #0D1525;
    border: 1px solid #1E2A40;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.gas-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.75rem;
}
.result-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #131D2E;
    font-size: 0.78rem;
}
.result-item:last-child { border-bottom: none; }
.result-text {
    flex: 1;
    color: #CBD5E1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-family: 'DM Mono', monospace;
    font-size: 0.73rem;
}
.badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    padding: 3px 9px;
    border-radius: 5px;
    white-space: nowrap;
    flex-shrink: 0;
}
.badge-gas  { background: #2D1E08; color: #F59E0B; border: 1px solid #78350F; }
.badge-safe { background: #052E1B; color: #34D399; border: 1px solid #064E3B; }
.badge-unc  { background: #131D2E; color: #64748B; border: 1px solid #1E2A40; }
.badge-hc   { background: #1A1A3E; color: #818CF8; border: 1px solid #312E81; }
.bar-wrap {
    width: 90px;
    height: 5px;
    background: #131D2E;
    border-radius: 3px;
    overflow: hidden;
    flex-shrink: 0;
}
.bar-fill { height: 100%; border-radius: 3px; }
.conf-pct {
    font-size: 0.68rem;
    color: #64748B;
    min-width: 34px;
    text-align: right;
    flex-shrink: 0;
}
.info-banner {
    background: #0A1628;
    border: 1px solid #1E3A5F;
    border-left: 3px solid #3B82F6;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.75rem;
    color: #93C5FD;
    margin-bottom: 1.25rem;
}
[data-testid="stTextArea"] textarea {
    background: #0D1525 !important;
    border: 1px solid #1E2A40 !important;
    border-radius: 8px !important;
    color: #CBD5E1 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}
.stButton > button {
    background: #111827 !important;
    border: 1px solid #1E2A40 !important;
    border-radius: 8px !important;
    color: #CBD5E1 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #1E2A40 !important;
    border-color: #334155 !important;
    color: #F8FAFC !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #64748B !important;
}
[data-testid="stTabs"] [aria-selected="true"] { color: #F8FAFC !important; }
hr { border-color: #1E2A40 !important; margin: 1rem 0 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #080C14; }
::-webkit-scrollbar-thumb { background: #1E2A40; border-radius: 3px; }
#MainMenu, footer, header { visibility: hidden; }
"""

st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)



def capture_stdout(fn, *args, **kwargs):
    """Run fn(*args, **kwargs), capture stdout, return (result, log_str)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


@st.cache_resource(show_spinner=False)
def _cached_load():
    models, _ = capture_stdout(load_or_train)
    return models


def get_models():
    if "models" in st.session_state and st.session_state["models"] is not None:
        return st.session_state["models"]
    return _cached_load()


def invalidate_model_cache():
    st.session_state.pop("models", None)
    _cached_load.clear()


def load_training_log(model_dir: str = MODEL_DIR):
    path = os.path.join(model_dir, "training_log.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def conf_color(label: str) -> str:
    return {"GASLIGHTING": "#F59E0B", "NOT GASLIGHTING": "#10B981"}.get(label, "#475569")


def result_rows_html(df: pd.DataFrame) -> str:
    html = ""
    for _, row in df.iterrows():
        txt   = str(row["text"])[:95].replace("<", "&lt;").replace(">", "&gt;")
        lbl   = row["consensus_label"]
        conf  = float(row["consensus_confidence"])
        hc    = bool(row["is_high_confidence"])
        pct   = int(conf * 100)
        color = conf_color(lbl)

        if lbl == "GASLIGHTING":
            badge = '<span class="badge badge-gas">GASLIGHTING</span>'
        elif lbl == "NOT GASLIGHTING":
            badge = '<span class="badge badge-safe">NOT GASLIGHTING</span>'
        else:
            badge = '<span class="badge badge-unc">UNCERTAIN</span>'

        hc_badge = '<span class="badge badge-hc">★ HC</span>' if hc else ""

        html += (
            f'<div class="result-item">'
            f'{badge}{hc_badge}'
            f'<span class="result-text" title="{txt}">{txt}</span>'
            f'<div class="bar-wrap"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'<span class="conf-pct">{pct}%</span>'
            f'</div>'
        )
    return html



with st.sidebar:
    st.markdown(
        '<p style="font-family:\'Syne\',sans-serif;font-size:1rem;font-weight:700;'
        'letter-spacing:-0.02em;color:#F8FAFC;margin-bottom:0">🔍 That Never Happened</p>'
        '<p style="font-size:0.65rem;color:#334155;margin-top:2px;margin-bottom:1.5rem">'
        'Gaslighting detection</p>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "nav",
        ["Detect", "Train", "History"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Model status
    if PROJECT_AVAILABLE:
        if models_exist():
            log = load_training_log()
            if log:
                last = log[-1]
                st.markdown(
                    f'<div style="font-size:0.7rem;color:#475569;line-height:1.8">'
                    f'<span style="color:#10B981">●</span> Models ready<br>'
                    f'Iteration&nbsp;&nbsp;#{last["iteration"]}<br>'
                    f'LR acc&nbsp;&nbsp;&nbsp;&nbsp;{last["lr_accuracy"]:.3f}<br>'
                    f'LR F1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{last["lr_f1"]:.3f}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<span style="color:#10B981;font-size:0.7rem">● Models ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#F59E0B;font-size:0.7rem">● No saved models — train first</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#EF4444;font-size:0.7rem">● Import error</span>', unsafe_allow_html=True)

    if os.path.exists(ACCUM_LABELS_PATH):
        try:
            n = len(pd.read_csv(ACCUM_LABELS_PATH))
            st.markdown(
                f'<div style="margin-top:10px;font-size:0.68rem;color:#334155">'
                f'Pool: {n} accumulated labels</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass



if not PROJECT_AVAILABLE:
    st.error(f"Could not import project files: {_IMPORT_ERROR}")
    st.info(
        "Make sure `gaslighting_detector.py` and `gaslighting_data.py` "
        "are in the same directory as `app.py`."
    )
    st.stop()




if page == "Detect":

    st.markdown("<h1>Detect</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#475569;font-size:0.8rem;margin-bottom:1.5rem;'
        'font-family:\'DM Mono\',monospace">'
        'Classify conversation snippets — one per line or via file upload.</p>',
        unsafe_allow_html=True,
    )

    tab_text, tab_file = st.tabs(["  ✎ Free text  ", "  ↑ Upload file  "])
    texts_to_classify: list[str] = []

    with tab_text:
        raw = st.text_area(
            "conversations",
            placeholder=(
                "You're overreacting, that never happened.\n"
                "I understand your point, even if I see it differently.\n"
                "Nobody else thinks it went that way."
            ),
            height=130,
            label_visibility="collapsed",
        )
        if raw.strip():
            texts_to_classify = [t.strip() for t in raw.splitlines() if t.strip()]

    with tab_file:
        uploaded = st.file_uploader(
            "upload",
            type=["txt", "csv"],
            label_visibility="collapsed",
        )
        if uploaded:
            if uploaded.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded)
                col = st.selectbox("Text column", df_up.columns.tolist())
                texts_to_classify = df_up[col].dropna().astype(str).tolist()
                st.caption(f"{len(texts_to_classify)} rows loaded.")
            else:
                texts_to_classify = [
                    l.strip()
                    for l in uploaded.read().decode("utf-8").splitlines()
                    if l.strip()
                ]
                st.caption(f"{len(texts_to_classify)} lines loaded.")

    col_thr, col_btn, _ = st.columns([2, 1, 4])
    with col_thr:
        threshold = st.slider(
            "Confidence threshold",
            min_value=0.50, max_value=0.99,
            value=0.80, step=0.01, format="%.2f",
        )
    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run = st.button("▶  Analyse", use_container_width=True)

    if run:
        if not texts_to_classify:
            st.warning("Add at least one conversation snippet.")
        elif not models_exist():
            st.error("No saved models found. Go to **Train** first.")
        else:
            with st.spinner("Classifying …"):
                models = get_models()
                rdf, _ = capture_stdout(
                    predict_conversations,
                    texts_to_classify,
                    models,
                    confidence_threshold=threshold,
                    save_path=None,
                )
            st.session_state["results_df"] = rdf

    if st.session_state.get("results_df") is not None:
        rdf: pd.DataFrame = st.session_state["results_df"]

        n_gas  = (rdf["consensus_label"] == "GASLIGHTING").sum()
        n_safe = (rdf["consensus_label"] == "NOT GASLIGHTING").sum()
        n_unc  = (rdf["consensus_label"] == "UNCERTAIN").sum()
        n_hc   = int(rdf["is_high_confidence"].sum())

        st.markdown(
            f'<div class="metric-grid">'
            f'<div class="metric-box"><div class="mlabel">Total</div>'
            f'<div class="mvalue">{len(rdf)}</div></div>'
            f'<div class="metric-box"><div class="mlabel">Gaslighting</div>'
            f'<div class="mvalue gas">{n_gas}</div></div>'
            f'<div class="metric-box"><div class="mlabel">Not gaslighting</div>'
            f'<div class="mvalue safe">{n_safe}</div></div>'
            f'<div class="metric-box"><div class="mlabel">High confidence</div>'
            f'<div class="mvalue hc">{n_hc}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="gas-card">'
            f'<div class="gas-card-title">Classification results</div>'
            f'{result_rows_html(rdf)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_dl, col_pool, _ = st.columns([2, 3, 3])
        with col_dl:
            st.download_button(
                "↓ Export CSV",
                data=rdf.to_csv(index=False).encode(),
                file_name="gaslighting_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_pool:
            hc_rows = rdf[rdf["is_high_confidence"]]
            if not hc_rows.empty:
                if st.button(f"＋ Add {n_hc} HC rows to training pool", use_container_width=True):
                    save_df = hc_rows.copy()
                    save_df["label"] = save_df["consensus_label"].map(
                        {"GASLIGHTING": 1, "NOT GASLIGHTING": 0}
                    )
                    save_df["timestamp"] = datetime.now().isoformat(timespec="seconds")
                    append_labeled_data(
                        save_df[["text", "label", "consensus_confidence", "timestamp"]],
                        path=ACCUM_LABELS_PATH,
                    )
                    st.success(f"{n_hc} rows saved → `{ACCUM_LABELS_PATH}`.")
            else:
                st.caption("No high-confidence rows to add.")



elif page == "Train":

    st.markdown("<h1>Train</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#475569;font-size:0.8rem;margin-bottom:1.5rem;'
        'font-family:\'DM Mono\',monospace">'
        'Initial training on seed data, or retrain on accumulated labels.</p>',
        unsafe_allow_html=True,
    )

    # Banner
    if os.path.exists(ACCUM_LABELS_PATH):
        try:
            n_pool = len(pd.read_csv(ACCUM_LABELS_PATH))
            log    = load_training_log()
            since  = f" · <strong>{n_pool} labels in pool</strong>" if n_pool else ""
            st.markdown(
                f'<div class="info-banner">'
                f'Accumulated labels available{since}. '
                f'{"Retraining recommended." if n_pool > 0 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass

    # SSL knobs
    st.markdown('<div class="gas-card-title">SSL settings</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        ssl_thr = st.slider("Pseudo-label threshold", 0.60, 0.99, 0.85, 0.01, format="%.2f")
    with c2:
        ssl_iter = st.slider("Max SSL iterations", 1, 10, 5)

    st.markdown("---")

    # Action cards
    col_init, col_ret = st.columns(2)

    with col_init:
        st.markdown(
            '<div class="gas-card">'
            '<div class="gas-card-title">Initial train</div>'
            '<p style="font-size:0.75rem;color:#475569;margin-bottom:1rem">'
            'Train from scratch on seed data. Forces a full retrain even if models exist.</p>',
            unsafe_allow_html=True,
        )
        if st.button("▶  Run initial train", key="btn_init", use_container_width=True):
            with st.spinner("Training — this may take a few minutes …"):
                try:
                    models, logs = capture_stdout(train, force=True)
                    invalidate_model_cache()
                    st.session_state["models"] = models
                    st.success("Training complete. Models saved.")
                    with st.expander("Show training output"):
                        st.code(logs, language=None)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_ret:
        st.markdown(
            '<div class="gas-card">'
            '<div class="gas-card-title">Retrain</div>'
            '<p style="font-size:0.75rem;color:#475569;margin-bottom:1rem">'
            'Retrain on seed data + all accumulated high-confidence predictions.</p>',
            unsafe_allow_html=True,
        )
        if st.button("↻  Run retrain", key="btn_retrain", use_container_width=True):
            if not os.path.exists(ACCUM_LABELS_PATH):
                st.warning("No accumulated labels yet — run predictions and add HC rows first.")
            else:
                with st.spinner("Retraining — this may take a few minutes …"):
                    try:
                        models, logs = capture_stdout(retrain)
                        invalidate_model_cache()
                        st.session_state["models"] = models
                        st.success("Retraining complete. Models updated.")
                        with st.expander("Show training output"):
                            st.code(logs, language=None)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Retraining failed: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Saved plots
    st.markdown("---")
    pc, pf = st.columns(2)
    cm_path   = os.path.join(MODEL_DIR, "confusion_matrices.png")
    feat_path = os.path.join(MODEL_DIR, "top_features.png")

    with pc:
        st.markdown('<div class="gas-card-title">Confusion matrix</div>', unsafe_allow_html=True)
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.caption("Run initial train to generate this plot.")

    with pf:
        st.markdown('<div class="gas-card-title">Top LR features</div>', unsafe_allow_html=True)
        if os.path.exists(feat_path):
            st.image(feat_path, use_container_width=True)
        else:
            st.caption("Run initial train to generate this plot.")




elif page == "History":

    st.markdown("<h1>History</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#475569;font-size:0.8rem;margin-bottom:1.5rem;'
        'font-family:\'DM Mono\',monospace">'
        'Model performance across training iterations.</p>',
        unsafe_allow_html=True,
    )

    history = load_training_log()

    if not history:
        st.info("No training history yet. Run a training cycle to populate this page.")
        st.stop()

    last = history[-1]

    # Summary metrics
    st.markdown(
        f'<div class="metric-grid">'
        f'<div class="metric-box"><div class="mlabel">Iterations</div>'
        f'<div class="mvalue">{len(history)}</div></div>'
        f'<div class="metric-box"><div class="mlabel">LR accuracy</div>'
        f'<div class="mvalue safe">{last["lr_accuracy"]:.3f}</div></div>'
        f'<div class="metric-box"><div class="mlabel">LR F1</div>'
        f'<div class="mvalue safe">{last["lr_f1"]:.3f}</div></div>'
        f'<div class="metric-box"><div class="mlabel">Train samples</div>'
        f'<div class="mvalue">{last["n_train"]}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Plotly chart
    iters     = [e["iteration"]      for e in history]
    lr_acc    = [e["lr_accuracy"]    for e in history]
    lr_f1     = [e["lr_f1"]          for e in history]
    sbert_acc = [e["sbert_accuracy"] for e in history]
    sbert_f1  = [e["sbert_f1"]       for e in history]
    n_trains  = [e["n_train"]        for e in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iters, y=lr_acc, name="LR accuracy",
        mode="lines+markers", line=dict(color="#10B981", width=2), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=iters, y=lr_f1,  name="LR F1",
        mode="lines+markers", line=dict(color="#10B981", width=2, dash="dot"), marker=dict(size=7)))

    if any(v > 0 for v in sbert_acc):
        fig.add_trace(go.Scatter(x=iters, y=sbert_acc, name="SBERT accuracy",
            mode="lines+markers", line=dict(color="#818CF8", width=2), marker=dict(size=7)))
        fig.add_trace(go.Scatter(x=iters, y=sbert_f1, name="SBERT F1",
            mode="lines+markers", line=dict(color="#818CF8", width=2, dash="dot"), marker=dict(size=7)))

    for i, (it, n) in enumerate(zip(iters, n_trains)):
        fig.add_annotation(
            x=it, y=max(lr_acc[i], lr_f1[i]) + 0.025,
            text=f"n={n}", showarrow=False,
            font=dict(size=10, color="#475569"),
        )

    fig.update_layout(
        paper_bgcolor="#080C14",
        plot_bgcolor="#0D1525",
        font=dict(family="DM Mono", color="#94A3B8", size=11),
        xaxis=dict(
            title="Iteration", tickmode="array", tickvals=iters,
            gridcolor="#131D2E", linecolor="#1E2A40",
        ),
        yaxis=dict(
            title="Score", range=[0.4, 1.05],
            gridcolor="#131D2E", linecolor="#1E2A40",
        ),
        legend=dict(bgcolor="#0D1525", bordercolor="#1E2A40", borderwidth=1),
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Log table
    st.markdown('<div class="gas-card-title">Iteration log</div>', unsafe_allow_html=True)
    log_df = pd.DataFrame(history)[
        ["iteration", "timestamp", "n_train", "n_test",
         "lr_accuracy", "lr_f1", "sbert_accuracy", "sbert_f1"]
    ].rename(columns={
        "iteration": "#", "timestamp": "date",
        "n_train": "n train", "n_test": "n test",
        "lr_accuracy": "LR acc", "lr_f1": "LR F1",
        "sbert_accuracy": "SBERT acc", "sbert_f1": "SBERT F1",
    })

    st.dataframe(
        log_df.sort_values("#", ascending=False).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "↓ Export training log (JSON)",
        data=json.dumps(history, indent=2),
        file_name="training_log.json",
        mime="application/json",
    )