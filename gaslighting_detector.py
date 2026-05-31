"""
 "That Never Happened": Detecting Gaslighting in Text

 Models  : Logistic Regression (TF-IDF + linguistic)
           SBERT + MLP         (transformer-based)

 Workflow
 --------
 1. TRAIN   — train both models on seed data, save to disk.
 2. PREDICT — load saved models, classify new conversations,
              save results to accumulated_labels.csv.
 3. RETRAIN — reload accumulated data, retrain both models,
              save updated versions.  Repeat from step 2.

 Files written
 ─────────────
 saved_models/
   lr_model.pkl          Logistic Regression
   tfidf_vectorizer.pkl  fitted TF-IDF
   sbert_mlp.pkl         MLP on top of SBERT embeddings
   ling_scaler.pkl       StandardScaler for linguistic features
   training_log.json     iteration history (date, metrics, counts)
 accumulated_labels.csv  grows with each predict → label cycle
"""

# 0.  IMPORTS

import os
import re
import json
import string
import warnings
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)
from scipy.sparse import hstack, vstack as sp_vstack, csr_matrix, issparse

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

for _r in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    nltk.download(_r, quiet=True)

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("[WARN] sentence-transformers not installed — SBERT model skipped.")
    print("       pip install sentence-transformers\n")

from gaslighting_data import (
    load_data, load_unlabeled, append_labeled_data,
    load_accumulated_data, REDDIT_CSV_PATH,
)


# PATHS & CONSTANTS

MODEL_DIR          = "saved_models"
ACCUM_LABELS_PATH  = "accumulated_labels.csv"
SBERT_MODEL_NAME   = "all-MiniLM-L6-v2"

# Artifact filenames inside MODEL_DIR
_LR_FILE      = "lr_model.pkl"
_TFIDF_FILE   = "tfidf_vectorizer.pkl"
_MLP_FILE     = "sbert_mlp.pkl"
_SCALER_FILE  = "ling_scaler.pkl"
_LOG_FILE     = "training_log.json"


# 1.  TEXT PREPROCESSING

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))
_stop_words -= {"you", "your", "yourself"}   # keep 2nd-person pronouns

_CONTRACTIONS = {
    "you're":"you are","you've":"you have","you'll":"you will",
    "you'd":"you would","i'm":"i am","i've":"i have","i'll":"i will",
    "i'd":"i would","that's":"that is","it's":"it is","can't":"cannot",
    "won't":"will not","don't":"do not","doesn't":"does not",
    "didn't":"did not","haven't":"have not","isn't":"is not",
    "aren't":"are not","wasn't":"was not","weren't":"were not",
    "there's":"there is","they're":"they are","needn't":"need not",
}


def preprocess(text: str) -> str:
    """Lowercase → expand contractions → strip punctuation → tokenise
    → remove stopwords → lemmatise."""
    text = text.lower()
    for c, e in _CONTRACTIONS.items():
        text = text.replace(c, e)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [_lemmatizer.lemmatize(t)
              for t in tokens if t not in _stop_words and t.isalpha()]
    return " ".join(tokens)


def preprocess_series(series: pd.Series) -> pd.Series:
    return series.apply(preprocess)


# 2.  LINGUISTIC & PRAGMATIC FEATURES

EPISTEMIC_VERBS = [
    "imagine","imagining","imagined","remember","remembering",
    "misremember","misremembering","misheard","confused","confuse",
    "hallucinate","hallucinating","dreamed","dreamt","perceive",
]

REALITY_DENIAL_PHRASES = [
    "never happened","that did not happen","that didn't happen",
    "making things up","making it up","made that up",
    "you imagined","imagining things","living in your head",
    "overreacting","too sensitive","being dramatic","so dramatic",
    "being irrational","not thinking clearly","can't trust yourself",
    "can't trust your","your memory","misremembering",
    "put words in","twist my words","twisting my words",
    "you always","you never","trust me on this",
    "nobody else","everyone agrees","only one who",
    "just in your head","blowing out of proportion",
    "paranoid","delusional","crazy if you",
]

SECOND_PERSON_TOKENS = {"you", "your", "yourself", "youre", "you are"}

LING_FEATURE_NAMES = [
    "second_person_count", "epistemic_verb_count",
    "reality_denial_count", "sentiment_polarity",
    "sentiment_subjectivity", "polarity_mismatch",
    "word_count", "you_density",
]


def build_linguistic_matrix(texts: pd.Series) -> np.ndarray:
    """Return (n_samples, 8) hand-crafted feature matrix."""
    rows = []
    for text in texts:
        wc   = max(len(text.split()), 1)
        sp   = sum(1 for t in re.findall(r"\b\w+\b", text.lower())
                   if t in SECOND_PERSON_TOKENS)
        ep   = sum(1 for v in EPISTEMIC_VERBS if v in text.lower())
        rd   = sum(1 for p in REALITY_DENIAL_PHRASES if p in text.lower())
        blob = TextBlob(text)
        pol, sub = blob.sentiment.polarity, blob.sentiment.subjectivity
        mm   = -pol * sub if pol < 0 else 0.0
        rows.append([sp, ep, rd, pol, sub, mm, wc, sp / wc])
    return np.array(rows, dtype=float)


# 3.  FEATURE MATRIX BUILDERS

def build_tfidf_ling(clean: pd.Series, raw: pd.Series,
                     tfidf: TfidfVectorizer, fit: bool = True):
    """TF-IDF unigrams+bigrams stacked with 8 linguistic features."""
    mat = tfidf.fit_transform(clean) if fit else tfidf.transform(clean)
    return hstack([mat, csr_matrix(build_linguistic_matrix(raw))])


def build_sbert_ling(raw: pd.Series, sbert,
                     scaler: StandardScaler, fit: bool = True) -> np.ndarray:
    """SBERT 384-dim embeddings stacked with scaled linguistic features."""
    emb  = sbert.encode(raw.tolist(), batch_size=32,
                        show_progress_bar=False, convert_to_numpy=True)
    ling = build_linguistic_matrix(raw)
    ling = scaler.fit_transform(ling) if fit else scaler.transform(ling)
    return np.hstack([emb, ling])


# 4.  MODEL PERSISTENCE  ← NEW

def models_exist(model_dir: str = MODEL_DIR) -> bool:
    """Return True only if every required artifact is present on disk."""
    required = [_LR_FILE, _TFIDF_FILE, _MLP_FILE, _SCALER_FILE]
    return all(os.path.exists(os.path.join(model_dir, f)) for f in required)


def save_models(lr, tfidf, mlp, scaler, metrics: dict,
                model_dir: str = MODEL_DIR) -> None:
    """
    Persist all model artifacts and append an entry to the training log.

    Saves
    -----
    lr_model.pkl          — Logistic Regression
    tfidf_vectorizer.pkl  — fitted TF-IDF vectorizer
    sbert_mlp.pkl         — MLP classifier
    ling_scaler.pkl       — StandardScaler for linguistic features
    training_log.json     — appended entry: date, iteration, metrics
    """
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(lr,     os.path.join(model_dir, _LR_FILE))
    joblib.dump(tfidf,  os.path.join(model_dir, _TFIDF_FILE))
    joblib.dump(scaler, os.path.join(model_dir, _SCALER_FILE))

    if mlp is not None:
        joblib.dump(mlp, os.path.join(model_dir, _MLP_FILE))

    # Update training log
    log_path = os.path.join(model_dir, _LOG_FILE)
    history  = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            history = json.load(f)

    entry = {
        "iteration"    : len(history) + 1,
        "timestamp"    : datetime.now().isoformat(timespec="seconds"),
        "n_train"      : metrics.get("n_train", "?"),
        "n_test"       : metrics.get("n_test",  "?"),
        "lr_accuracy"  : round(metrics.get("lr_accuracy",  0), 4),
        "lr_f1"        : round(metrics.get("lr_f1",        0), 4),
        "sbert_accuracy": round(metrics.get("sbert_accuracy", 0), 4),
        "sbert_f1"     : round(metrics.get("sbert_f1",     0), 4),
    }
    history.append(entry)

    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[SAVE] Models saved to '{model_dir}/'")
    print(f"       Iteration #{entry['iteration']} logged  "
          f"({entry['timestamp']})")
    print(f"       LR  — Acc={entry['lr_accuracy']:.4f}  "
          f"F1={entry['lr_f1']:.4f}")
    if mlp is not None:
        print(f"       SBERT — Acc={entry['sbert_accuracy']:.4f}  "
              f"F1={entry['sbert_f1']:.4f}")


def load_models(model_dir: str = MODEL_DIR) -> dict | None:
    """
    Load all artifacts from disk.

    Returns
    -------
    dict with keys: lr, tfidf, mlp (None if SBERT unavailable), scaler, log
    Returns None if any required file is missing.
    """
    if not models_exist(model_dir):
        return None

    lr     = joblib.load(os.path.join(model_dir, _LR_FILE))
    tfidf  = joblib.load(os.path.join(model_dir, _TFIDF_FILE))
    scaler = joblib.load(os.path.join(model_dir, _SCALER_FILE))

    mlp = None
    mlp_path = os.path.join(model_dir, _MLP_FILE)
    if os.path.exists(mlp_path):
        mlp = joblib.load(mlp_path)

    log = []
    log_path = os.path.join(model_dir, _LOG_FILE)
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)

    last = log[-1] if log else {}
    print(f"\n[LOAD] Models loaded from '{model_dir}/'")
    if last:
        print(f"       Iteration #{last['iteration']}  "
              f"trained {last['timestamp']}")
        print(f"       LR  — Acc={last['lr_accuracy']}  F1={last['lr_f1']}")
        if mlp:
            print(f"       SBERT — Acc={last['sbert_accuracy']}  "
                  f"F1={last['sbert_f1']}")

    return {"lr": lr, "tfidf": tfidf, "mlp": mlp, "scaler": scaler, "log": log}


def print_training_history(model_dir: str = MODEL_DIR) -> None:
    """Pretty-print the full training log."""
    log_path = os.path.join(model_dir, _LOG_FILE)
    if not os.path.exists(log_path):
        print("[LOG] No training history found.")
        return
    with open(log_path) as f:
        history = json.load(f)

    print(f"\n{'='*70}")
    print(f"  Training History  ({len(history)} iteration(s))")
    print(f"{'='*70}")
    print(f"  {'#':<4} {'Date':<22} {'N_train':>8} {'LR_Acc':>8} "
          f"{'LR_F1':>8} {'SB_Acc':>8} {'SB_F1':>8}")
    print("  " + "─"*64)
    for e in history:
        print(f"  {e['iteration']:<4} {e['timestamp']:<22} "
              f"{str(e['n_train']):>8} {e['lr_accuracy']:>8.4f} "
              f"{e['lr_f1']:>8.4f} {e['sbert_accuracy']:>8.4f} "
              f"{e['sbert_f1']:>8.4f}")


# 5.  EVALUATION

def evaluate_model(model, X_test, y_test,
                   name: str, train_time: float = 0.0) -> dict:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {pre:.4f}  (gaslighting class)")
    print(f"  Recall    : {rec:.4f}  (gaslighting class)")
    print(f"  F1-Score  : {f1:.4f}  (gaslighting class)")
    print(f"  Train time: {train_time:.2f}s")
    print()
    print(classification_report(y_test, y_pred,
          target_names=["Not Gaslighting", "Gaslighting"]))
    print(f"  Confusion Matrix:\n{cm}\n")

    return dict(name=name, accuracy=acc, precision=pre,
                recall=rec, f1=f1, cm=cm, train_time=train_time)


# 6.  VISUALISATIONS

def plot_confusion_matrices(results: list, save_path: str) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        ConfusionMatrixDisplay(res["cm"],
            display_labels=["Not Gas.", "Gaslighting"]
        ).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{res['name']}\nAcc={res['accuracy']:.3f}",
                     fontsize=10, fontweight="bold")
    plt.suptitle("Confusion Matrices", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Confusion matrices → {save_path}")


def plot_top_features(lr, tfidf: TfidfVectorizer,
                      n: int = 20, save_path: str = "top_features.png") -> None:
    if not hasattr(lr, "coef_"):
        return
    all_names = tfidf.get_feature_names_out().tolist() + LING_FEATURE_NAMES
    coefs     = lr.coef_[0]
    if len(coefs) != len(all_names):
        return

    feat_df = (pd.DataFrame({"feature": all_names, "coefficient": coefs})
               .assign(abs_coef=lambda d: d["coefficient"].abs())
               .sort_values("abs_coef", ascending=False)
               .head(n).sort_values("coefficient"))

    colors = ["#bd4437" if c > 0 else "#3498db" for c in feat_df["coefficient"]]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(feat_df["feature"], feat_df["coefficient"], color=colors)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Coefficient", fontsize=12)
    ax.set_title(f"Top {n} LR Coefficients\n"
                 "(Red = Gaslighting, Blue = Non-gaslighting)",
                 fontsize=12, fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(color="#e74c3c", label="Gaslighting"),
        mpatches.Patch(color="#3498db", label="Not Gaslighting"),
    ], fontsize=9)
    ax.set_facecolor("#f9f9f9")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Feature importance → {save_path}")


def plot_training_progress(model_dir: str = MODEL_DIR,
                           save_path: str = "training_progress.png") -> None:
    """
    Plot LR and SBERT accuracy + F1 across training iterations.
    Gives a clear view of how the model improves as more data accumulates.
    """
    log_path = os.path.join(model_dir, _LOG_FILE)
    if not os.path.exists(log_path):
        print("[PLOT] No training log found — skipping progress chart.")
        return
    with open(log_path) as f:
        history = json.load(f)
    if len(history) < 2:
        print("[PLOT] Need at least 2 iterations to plot progress.")
        return

    iters      = [e["iteration"]      for e in history]
    lr_acc     = [e["lr_accuracy"]    for e in history]
    lr_f1      = [e["lr_f1"]          for e in history]
    sbert_acc  = [e["sbert_accuracy"] for e in history]
    sbert_f1   = [e["sbert_f1"]       for e in history]
    n_trains   = [e["n_train"]        for e in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Accuracy
    ax1.plot(iters, lr_acc,    "o-", color="#2ecc71", lw=2, label="LR Accuracy")
    ax1.plot(iters, sbert_acc, "s-", color="#3498db", lw=2, label="SBERT Accuracy")
    ax1.set_xlabel("Training Iteration", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Accuracy over Iterations", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_ylim(0.4, 1.05)
    ax1.set_xticks(iters)
    ax1.grid(alpha=0.3)
    ax1.set_facecolor("#f9f9f9")

    # F1
    ax2.plot(iters, lr_f1,    "o-", color="#e67e22", lw=2, label="LR F1")
    ax2.plot(iters, sbert_f1, "s-", color="#e74c3c", lw=2, label="SBERT F1")
    for i, (it, n) in enumerate(zip(iters, n_trains)):
        ax2.annotate(f"n={n}", (it, max(lr_f1[i], sbert_f1[i])),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color="gray")
    ax2.set_xlabel("Training Iteration", fontsize=11)
    ax2.set_ylabel("F1-Score (gaslighting class)", fontsize=11)
    ax2.set_title("F1-Score over Iterations\n(n = training samples)",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0.4, 1.05)
    ax2.set_xticks(iters)
    ax2.grid(alpha=0.3)
    ax2.set_facecolor("#fffefe")

    plt.suptitle('"That Never Happened" — Model Progress',
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Training progress → {save_path}")


# 7.  SEMI-SUPERVISED LOOP

def semi_supervised_loop(labeled_X, labeled_y, unlabeled_X,
                         base_model, threshold=0.85, max_iter=5) -> tuple:
    """
    Self-training pseudo-label loop.
    Returns (fitted_model, augmented_X, augmented_y).
    """
    from sklearn.base import clone

    model     = clone(base_model)
    cur_X     = labeled_X
    cur_y     = np.array(labeled_y)
    remaining = list(range(unlabeled_X.shape[0]))

    print(f"  Labelled: {len(cur_y)} | Unlabelled pool: {len(remaining)}")

    for it in range(1, max_iter + 1):
        model.fit(cur_X, cur_y)
        if not remaining or not hasattr(model, "predict_proba"):
            break

        subset  = unlabeled_X[np.array(remaining)]
        proba   = model.predict_proba(subset)
        max_p   = proba.max(axis=1)
        p_label = proba.argmax(axis=1)
        mask    = max_p >= threshold
        n_new   = mask.sum()

        if n_new == 0:
            print(f"  [iter {it}] No high-confidence samples — stopping.")
            break

        new_X     = subset[np.where(mask)[0]]
        cur_X     = sp_vstack([cur_X, new_X]) if issparse(cur_X) \
                    else np.vstack([cur_X, new_X])
        cur_y     = np.concatenate([cur_y, p_label[mask]])
        remaining = [remaining[i] for i, f in enumerate(mask) if not f]

        print(f"  [iter {it}]  +{n_new} pseudo-labels | "
              f"total: {len(cur_y)} | remaining: {len(remaining)}")

    model.fit(cur_X, cur_y)
    return model, cur_X, cur_y


# 8.  CORE TRAINING FUNCTION

def _train_all(df: pd.DataFrame, sbert=None) -> dict:
    """
    Internal helper: train LR and SBERT+MLP on df, return all artifacts.
    Does NOT save anything — saving is the caller's responsibility.
    """
    df["clean_text"] = preprocess_series(df["text"])

    Xc_tr, Xc_te, Xr_tr, Xr_te, y_tr, y_te = train_test_split(
        df["clean_text"], df["text"], df["label"],
        test_size=0.20, random_state=42, stratify=df["label"],
    )
    print(f"[SPLIT] Train: {len(y_tr)} | Test: {len(y_te)}")

    # Logistic Regression
    print("\n[TRAIN] Logistic Regression …")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5_000,
                            sublinear_tf=True, min_df=1)
    X_tr_t = build_tfidf_ling(Xc_tr, Xr_tr, tfidf, fit=True)
    X_te_t = build_tfidf_ling(Xc_te, Xr_te, tfidf, fit=False)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=42, C=1.0, solver="lbfgs")
    t0 = time.perf_counter()
    lr.fit(X_tr_t, y_tr)
    res_lr = evaluate_model(lr, X_te_t, y_te,
                            "Logistic Regression", time.perf_counter() - t0)

    # SBERT + MLP 
    mlp = scaler = None
    res_mlp = {"accuracy": 0, "f1": 0}

    if SBERT_AVAILABLE and sbert is not None:
        print("\n[TRAIN] SBERT + MLP …")
        scaler = StandardScaler()
        X_tr_s = build_sbert_ling(Xr_tr, sbert, scaler, fit=True)
        X_te_s = build_sbert_ling(Xr_te, sbert, scaler, fit=False)

        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                            max_iter=500, random_state=42,
                            early_stopping=True, validation_fraction=0.1,
                            n_iter_no_change=15, learning_rate_init=1e-3)
        t0 = time.perf_counter()
        mlp.fit(X_tr_s, y_tr)
        res_mlp = evaluate_model(mlp, X_te_s, y_te,
                                 "SBERT + MLP", time.perf_counter() - t0)

    metrics = {
        "n_train"       : len(y_tr),
        "n_test"        : len(y_te),
        "lr_accuracy"   : res_lr["accuracy"],
        "lr_f1"         : res_lr["f1"],
        "sbert_accuracy": res_mlp["accuracy"],
        "sbert_f1"      : res_mlp["f1"],
    }
    return dict(lr=lr, tfidf=tfidf, mlp=mlp, scaler=scaler,
                metrics=metrics, res_lr=res_lr, res_mlp=res_mlp,
                X_te_t=X_te_t, y_te=y_te)


# 9.  PUBLIC API — TRAIN, PREDICT, RETRAIN  ← NEW WORKFLOW

def train(csv_path: str = REDDIT_CSV_PATH,
          model_dir: str = MODEL_DIR,
          force: bool = False) -> dict:
    """
    STEP 1 — Initial training.

    Loads seed data, trains LR + SBERT+MLP, runs the SSL loop,
    saves all artifacts to model_dir.

    Parameters
    ----------
    csv_path  : path to casual_data_windows.csv (for Reddit non-gas examples)
    model_dir : where to save model artifacts
    force     : if True, retrain even if saved models already exist

    Returns
    -------
    dict with keys: lr, tfidf, mlp, scaler  (ready to use for inference)
    """
    if models_exist(model_dir) and not force:
        print(f"[TRAIN] Saved models found in '{model_dir}/'.")
        print("        Use load_or_train() to load them, or pass force=True to retrain.")
        return load_models(model_dir)

    print("\n" + "="*60)
    print('  "That Never Happened" — Initial Training')
    print("="*60)

    df        = load_data(csv_path=csv_path)
    unlabeled = load_unlabeled(csv_path=csv_path)
    print(f"\n[DATA] Labelled   : {len(df)} "
          f"({df['label'].sum()} gas, {(df['label']==0).sum()} not-gas)")
    print(f"[DATA] Unlabelled : {len(unlabeled)} (SSL pool)")

    sbert = SentenceTransformer(SBERT_MODEL_NAME) if SBERT_AVAILABLE else None

    artifacts = _train_all(df, sbert=sbert)

    # SSL loop on LR
    print("\n[SSL] Running semi-supervised loop on LR …")
    df["clean_text"] = preprocess_series(df["text"])
    Xc_tr, _, Xr_tr, _, y_tr, _ = train_test_split(
        df["clean_text"], df["text"], df["label"],
        test_size=0.20, random_state=42, stratify=df["label"],
    )
    # Re-use the already-fitted tfidf
    unl_clean = preprocess_series(unlabeled["text"])
    X_unl = build_tfidf_ling(unl_clean, unlabeled["text"],
                              artifacts["tfidf"], fit=False)
    X_tr_t = build_tfidf_ling(Xc_tr, Xr_tr, artifacts["tfidf"], fit=False)

    ssl_lr, _, _ = semi_supervised_loop(
        labeled_X=X_tr_t, labeled_y=y_tr.values,
        unlabeled_X=X_unl,
        base_model=LogisticRegression(max_iter=1000, class_weight="balanced",
                                      random_state=42),
        threshold=0.85, max_iter=5,
    )
    artifacts["lr"] = ssl_lr   # replace with SSL-enhanced version

    # Save
    save_models(artifacts["lr"], artifacts["tfidf"],
                artifacts["mlp"], artifacts["scaler"],
                artifacts["metrics"], model_dir)

    # Plots
    plot_confusion_matrices(
        [artifacts["res_lr"], artifacts["res_mlp"]],
        os.path.join(model_dir, "confusion_matrices.png"),
    ) if artifacts["res_mlp"]["accuracy"] > 0 else \
    plot_confusion_matrices(
        [artifacts["res_lr"]],
        os.path.join(model_dir, "confusion_matrices.png"),
    )
    plot_top_features(artifacts["lr"], artifacts["tfidf"],
                      save_path=os.path.join(model_dir, "top_features.png"))

    return {k: artifacts[k] for k in ("lr", "tfidf", "mlp", "scaler")}


def load_or_train(csv_path: str = REDDIT_CSV_PATH,
                  model_dir: str = MODEL_DIR) -> dict:
    """
    Load saved models if they exist; otherwise run train() from scratch.
    This is the recommended entry point for inference workflows.

    Returns
    -------
    dict with keys: lr, tfidf, mlp, scaler
    """
    if models_exist(model_dir):
        return load_models(model_dir)
    print("[INFO] No saved models found — training from scratch …")
    return train(csv_path=csv_path, model_dir=model_dir)


def predict_conversations(texts: list[str],
                          models: dict,
                          confidence_threshold: float = 0.80,
                          save_path: str | None = ACCUM_LABELS_PATH) -> pd.DataFrame:
    """
    STEP 2 — Classify new conversations with the saved models.

    Both models vote independently. A consensus label is produced when
    they agree; otherwise the result is marked 'UNCERTAIN'.
    High-confidence rows (both models ≥ threshold and in agreement)
    are flagged for automatic inclusion in the next retraining cycle.

    Parameters
    ----------
    texts                : list of raw conversation strings to classify
    models               : dict returned by load_or_train() / train()
    confidence_threshold : minimum per-model probability to trust a prediction
    save_path            : if given, appends results to this CSV file

    Returns
    -------
    pd.DataFrame with columns:
        text, lr_label, lr_confidence,
        sbert_label, sbert_confidence,
        consensus_label, consensus_confidence,
        is_high_confidence, timestamp
    """
    lr, tfidf, mlp, scaler = (models[k] for k in ("lr", "tfidf", "mlp", "scaler"))

    raw_s   = pd.Series(texts)
    clean_s = preprocess_series(raw_s)

    # LR predictions
    X_t       = build_tfidf_ling(clean_s, raw_s, tfidf, fit=False)
    lr_pred   = lr.predict(X_t)
    lr_prob   = lr.predict_proba(X_t).max(axis=1)

    # SBERT predictions
    sbert_pred  = np.full(len(texts), -1, dtype=int)
    sbert_prob  = np.zeros(len(texts))
    sbert = None

    if SBERT_AVAILABLE and mlp is not None and scaler is not None:
        sbert = SentenceTransformer(SBERT_MODEL_NAME)
        X_s   = build_sbert_ling(raw_s, sbert, scaler, fit=False)
        sbert_pred = mlp.predict(X_s)
        sbert_prob = mlp.predict_proba(X_s).max(axis=1)

    lm = {1: "GASLIGHTING", 0: "NOT GASLIGHTING", -1: "N/A"}

    rows = []
    for i, text in enumerate(texts):
        lr_lbl    = lm[lr_pred[i]]
        lr_conf   = float(lr_prob[i])
        sb_lbl    = lm[sbert_pred[i]]
        sb_conf   = float(sbert_prob[i])

        # Consensus — three-way branch:
        #   1. SBERT unavailable → trust LR directly.
        #   2. Both models agree → emit consensus (average confidence).
        #   3. Models disagree   → defer to the more-confident model;
        #      mark UNCERTAIN only when neither clears the threshold.
        if sbert_pred[i] == -1:
            # SBERT not available — rely on LR alone
            consensus_lbl  = lr_lbl
            consensus_conf = round(lr_conf, 4)
            high_conf      = lr_conf >= confidence_threshold
        elif lr_pred[i] == sbert_pred[i]:
            # Agreement — always emit a label; confidence is their average
            consensus_lbl  = lr_lbl
            consensus_conf = round((lr_conf + sb_conf) / 2, 4)
            high_conf      = consensus_conf >= confidence_threshold
        else:
            # Disagreement — winner is whichever model is more confident
            if lr_conf >= sb_conf:
                consensus_lbl, consensus_conf = lr_lbl, round(lr_conf, 4)
            else:
                consensus_lbl, consensus_conf = sb_lbl, round(sb_conf, 4)
            high_conf = max(lr_conf, sb_conf) >= confidence_threshold
            if not high_conf:
                consensus_lbl = "UNCERTAIN"

        rows.append({
            "text"                : text,
            "lr_label"            : lr_lbl,
            "lr_confidence"       : round(lr_conf, 4),
            "sbert_label"         : sb_lbl,
            "sbert_confidence"    : round(sb_conf, 4),
            "consensus_label"     : consensus_lbl,
            "consensus_confidence": consensus_conf,
            "is_high_confidence"  : high_conf,
            "timestamp"           : datetime.now().isoformat(timespec="seconds"),
        })

    result_df = pd.DataFrame(rows)

    # Print summary
    n_gas     = (result_df["consensus_label"] == "GASLIGHTING").sum()
    n_not     = (result_df["consensus_label"] == "NOT GASLIGHTING").sum()
    n_unc     = (result_df["consensus_label"] == "UNCERTAIN").sum()
    n_hc      = result_df["is_high_confidence"].sum()

    print(f"\n[PREDICT] {len(texts)} conversations classified:")
    print(f"   GASLIGHTING     : {n_gas}")
    print(f"   NOT GASLIGHTING : {n_not}")
    print(f"   UNCERTAIN        : {n_unc}")
    print(f"   High-confidence  : {n_hc}  (ready for retraining)")

    # Save to accumulated labels file
    if save_path and n_hc > 0:
        hc_rows = result_df[result_df["is_high_confidence"]].copy()
        hc_rows["label"] = hc_rows["consensus_label"].map(
            {"GASLIGHTING": 1, "NOT GASLIGHTING": 0}
        )
        append_labeled_data(
            hc_rows[["text", "label", "consensus_confidence", "timestamp"]],
            path=save_path,
        )
        print(f"   {n_hc} high-confidence rows appended → '{save_path}'")

    return result_df


def retrain(csv_path: str = REDDIT_CSV_PATH,
            accum_path: str = ACCUM_LABELS_PATH,
            model_dir: str = MODEL_DIR) -> dict:
    """
    STEP 3 — Retrain on seed data + all accumulated predictions.

    Loads original seed data and all previously saved predictions,
    retrains LR + SBERT+MLP, evaluates, and overwrites saved models
    with the new versions. Appends a new entry to training_log.json.

    Parameters
    ----------
    csv_path   : Reddit CSV for seed non-gaslighting examples
    accum_path : accumulated_labels.csv (grows with each predict cycle)
    model_dir  : where to save updated models

    Returns
    -------
    dict with keys: lr, tfidf, mlp, scaler
    """
    print("\n" + "="*60)
    print('  "That Never Happened" — Retraining')
    print("="*60)

    df = load_accumulated_data(accum_path=accum_path, csv_path=csv_path)
    print(f"\n[DATA] Combined dataset: {len(df)} samples "
          f"({df['label'].sum()} gas, {(df['label']==0).sum()} not-gas)")

    sbert = SentenceTransformer(SBERT_MODEL_NAME) if SBERT_AVAILABLE else None

    artifacts = _train_all(df, sbert=sbert)

    save_models(artifacts["lr"], artifacts["tfidf"],
                artifacts["mlp"], artifacts["scaler"],
                artifacts["metrics"], model_dir)

    plot_top_features(artifacts["lr"], artifacts["tfidf"],
                      save_path=os.path.join(model_dir, "top_features.png"))
    plot_training_progress(model_dir=model_dir,
                           save_path=os.path.join(model_dir,
                                                   "training_progress.png"))
    print_training_history(model_dir)

    return {k: artifacts[k] for k in ("lr", "tfidf", "mlp", "scaler")}


# 10. Main function

if __name__ == "__main__":
    import sys
    args    = sys.argv[1:]
    command = args[0].lower() if args else "train"

    if command == "retrain":
        retrain()

    elif command == "predict":
        if len(args) < 2:
            print("Usage: python gaslighting_detector.py predict \"text1\" \"text2\" ...")
            sys.exit(1)
        models = load_or_train()
        results = predict_conversations(args[1:], models)
        print(results[["text", "consensus_label",
                        "consensus_confidence", "is_high_confidence"]].to_string())

    else:
        # Default: train (or load if models already saved)
        force = command == "train"
        models = train(force=force) if force else load_or_train()

        # Demo predictions
        demo = [
            "Hey, how are you doing? Everything good?",
            "I am good and what about you?",
            "I am doing just fine. Where have you been lately?",
            "Yeha I have been to Germany for a vacation with my family. Did you go somewhere too?",
            "You're imagining things that aren't there.",
            "Nobody else thinks it happened that way.",
        ]
        print("\n" + "="*60)
        print("  DEMO PREDICTIONS")
        print("="*60)
        results = predict_conversations(demo, models, save_path=None)
        lm = {"GASLIGHTING":"Gaslighting", "NOT GASLIGHTING" : "Not gaslighting", "UNCERTAIN": "Uncertain"}
        for _, row in results.iterrows():
            icon = lm.get(row["consensus_label"], "❓")
            print(f"  {icon} [{row['consensus_label']:<18}] "
                  f"({row['consensus_confidence']:.0%})  \"{row['text']}\"")
            
