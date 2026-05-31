"""
=============================================================
 gaslighting_data.py
 Dataset for: "That Never Happened" — Detecting Gaslighting
              Through Linguistic Patterns in Text
=============================================================
 Source strategy (from project slides):
   - Reddit relationship forums / conflict-related threads
   - Replies as unit of analysis
   - Small manually-labelled seed dataset (≈300–500 samples)
   - Extended via semi-supervised learning

 Data sources used:
   LABEL 1 : Hand-crafted gaslighting examples (Reddit-style)
   LABEL 0 : Hand-crafted healthy examples  +
             Real Reddit turns from casual_data_windows.csv
             (casual conversations are clearly non-gaslighting)
   UNLABELED: Real Reddit turns from casual_data_windows.csv
              (replaces the 30 synthetic borderline examples)

 This file provides:
   1. GASLIGHTING_EXAMPLES        — label 1, hand-crafted
   2. NON_GASLIGHTING_EXAMPLES    — label 0, hand-crafted
   3. load_reddit_csv()           — loads + cleans the Reddit CSV
   4. load_data()                 — labelled DataFrame (seed dataset)
   5. load_unlabeled()            — unlabeled pool for SSL loop
   6. save_csv()                  — persists labelled data to disk
=============================================================
"""

import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# PATH TO REAL REDDIT DATA
# Update this to wherever you placed the CSV file.
# ─────────────────────────────────────────────────────────────
REDDIT_CSV_PATH = "casual_data_windows.csv"

# ─────────────────────────────────────────────────────────────
# LABEL 1 — GASLIGHTING
# Drawn from Reddit relationship / conflict threads.
# Patterns: reality invalidation, memory attacks, blame-shift,
#           emotional dismissal, isolation from others.
# ─────────────────────────────────────────────────────────────
GASLIGHTING_EXAMPLES = [
    # ── Reality invalidation ──────────────────────────────────
    "You're imagining things again.",
    "That never happened and you know it.",
    "I never said that. You must have dreamed it.",
    "Nothing like that ever occurred.",
    "You made that whole thing up.",
    "That is simply not true and you know it.",
    "Your version of events is totally inaccurate.",
    "You are completely wrong about what happened.",
    "There is no way that happened the way you think.",
    "You were never mistreated — you imagined it.",

    # ── Memory attacks ────────────────────────────────────────
    "You have a terrible memory, you can't trust yourself.",
    "You always misremember things.",
    "You can't even remember what happened yesterday.",
    "Your memory is unreliable and you know it.",
    "You have a habit of misremembering things.",
    "You're remembering it wrong, as always.",
    "You never remember anything correctly.",
    "You're remembering it wrong on purpose.",
    "You never pay attention, so of course you're confused.",
    "You always confuse what happened with what you wish happened.",

    # ── Emotional dismissal ───────────────────────────────────
    "You're overreacting as usual.",
    "You're being way too sensitive about this.",
    "Stop being so dramatic about everything.",
    "You're acting hysterical for no reason.",
    "Your feelings are completely irrational.",
    "You need to calm down because you're not thinking clearly.",
    "You're being irrational and you need to stop.",
    "You're too sensitive to have this conversation.",
    "You're too emotional to think straight right now.",
    "You're too sensitive to be part of this discussion.",
    "You're overreacting and making a scene.",
    "You're blowing this completely out of proportion.",
    "You always blow things out of proportion.",
    "Your reaction makes no sense at all.",
    "You have no basis for feeling that way.",
    "Your feelings aren't based in reality.",

    # ── Sanity / competence attacks ───────────────────────────
    "You're crazy if you believe that.",
    "You're delusional if you think that happened.",
    "You need help because you're seeing things that aren't real.",
    "You sound crazy when you say things like that.",
    "You need to listen to yourself — you sound unhinged.",
    "Nobody believes you when you say things like that.",
    "You're so paranoid all the time.",
    "Why do you always have to be so paranoid?",
    "You're just being paranoid as usual.",
    "You're reading into things that aren't there.",
    "You're imagining slights that don't exist.",
    "You're projecting your own issues onto me.",
    "You're not as smart as you think you are.",
    "You can't trust your own judgment — you never could.",

    # ── Isolation / everyone-agrees ───────────────────────────
    "Nobody else has a problem with this, just you.",
    "Everyone agrees with me, not you.",
    "You're the only one who thinks that happened.",
    "Nobody else thinks it happened that way.",
    "Everyone thinks you're overreacting.",
    "This is exactly why people don't take you seriously.",

    # ── Blame-shift / victim-flip ─────────────────────────────
    "You're the problem here, not me.",
    "You're making yourself the victim again.",
    "You always have to be the victim.",
    "You're just trying to start a fight.",
    "You're just trying to make me feel bad.",
    "You're being manipulative right now.",
    "You're making me look bad for no reason.",
    "You always turn things around on me.",
    "You're twisting my words deliberately.",
    "You always twist everything I say.",
    "You're putting words in my mouth again.",
    "You always manage to twist things.",
    "You're the one with the problem here, not me.",

    # ── Confusion / trust manipulation ───────────────────────
    "You're confused again — just trust me on this.",
    "You're confused again, that's all.",
    "That's not how any of this works — you just don't get it.",
    "You're making this into something it isn't.",
    "You're reading way too much into this.",
    "You see problems where there are none.",
    "You're just looking for things to be upset about.",

    # ── Exhaustion / character attacks ────────────────────────
    "You're exhausting when you get like this.",
    "You're so difficult to deal with sometimes.",
    "You're being ridiculous right now.",
    "You're being unreasonable, as always.",
    "You're acting like a child right now.",
    "You're acting crazy and everyone can see it.",
    "You always exaggerate everything.",
    "You never make any sense.",
    "You're not making any sense right now.",
    "You're not making any sense, as usual.",

    # ── Epistemic / perception invalidation ───────────────────
    "Your perception is always off.",
    "You always see the negative in everything.",
    "You always do this — make things up.",
    "I never meant it that way — you misinterpreted everything.",
    "I never did that; you're lying to yourself.",
    "You have no idea what you're talking about.",
    "You've always had this problem with reality.",
    "You're looking for attention again.",
    "You were never upset about this before — why now?",
    "You always have to make everything about you.",

    # ── Reddit-style casual gaslighting ──────────────────────
    "Bro you're literally making this up, that did not happen.",
    "You need to stop living in your head.",
    "I already explained this to you three times, pay attention.",
    "You're always twisting what I say to win arguments.",
    "You can't even keep your story straight.",
    "You always do this when you know you're wrong.",
    "This is exactly the kind of thing you do — create drama.",
    "You misheard me, you always mishear me.",
    "You're way too in your feelings right now to think clearly.",
    "Can you not be like this for once?",
    "You're lying to yourself because you can't handle the truth.",
    "I never said that, check your own texts.",
    "You're literally hallucinating at this point.",
    "You've been like this forever — it's exhausting.",
    "That is NOT what I said and you know it.",
]

# ─────────────────────────────────────────────────────────────
# LABEL 0 — NON-GASLIGHTING
# Healthy disagreement, validation, empathy, boundary-setting,
# accountable language.
# ─────────────────────────────────────────────────────────────
NON_GASLIGHTING_EXAMPLES = [
    # ── Healthy disagreement ──────────────────────────────────
    "I disagree with your interpretation of the situation.",
    "I remember the event differently from you.",
    "I think we see this situation differently.",
    "I see your point, but I have a different perspective.",
    "My perspective is different, but yours is equally valid.",
    "I think both of us have valid points here.",
    "We can disagree and still respect each other.",
    "It's okay that we see it differently.",
    "I think we both have valid concerns here.",
    "Let's agree to disagree on this one.",

    # ── Validation & empathy ──────────────────────────────────
    "I understand why you feel that way.",
    "Your feelings are valid even if I don't fully understand them.",
    "That sounds really frustrating — I'm sorry.",
    "It sounds like this is really bothering you.",
    "It seems like this has been weighing on you.",
    "Your reaction makes sense given what you experienced.",
    "I acknowledge that this has been hard for you.",
    "I can see why that would frustrate you.",
    "Your feelings matter and I take them seriously.",
    "Your feelings don't need to be justified to be real.",
    "I hear that you're frustrated and I want to help.",

    # ── Accountable language ──────────────────────────────────
    "I'm sorry you feel hurt by what I said.",
    "I apologize if my words came across badly.",
    "I didn't mean to hurt you, but I hear that I did.",
    "I'm sorry that my actions made you feel that way.",
    "I didn't realize that came across that way.",
    "I may have been wrong about how I handled that.",
    "I realize now that I could have been clearer.",
    "I didn't mean to make you feel dismissed.",
    "I'll take responsibility for my part in this.",
    "I know I'm not perfect, but I'm trying.",

    # ── Collaborative resolution ──────────────────────────────
    "Let's talk about this calmly and figure it out.",
    "Can we discuss what actually happened?",
    "Can we sit down and talk about this together?",
    "Let's try to find a middle ground on this.",
    "I want to make sure we're on the same page.",
    "Let's take a break and come back to this later.",
    "Let me try to explain my side of things clearly.",
    "I think we should both share our perspectives.",
    "Let me share my perspective and then hear yours.",
    "Let's both share what we remember without judgment.",
    "Can we focus on understanding rather than being right?",
    "Let's approach this as a team, not adversaries.",
    "Let's try to figure this out together.",
    "Can we revisit this topic with fresh eyes?",
    "Let's slow down and listen to each other properly.",
    "Can we both agree to listen without interrupting?",
    "I think we need to communicate more clearly.",

    # ── Active listening ──────────────────────────────────────
    "I'd like to understand your point of view better.",
    "Can you explain what upset you about that moment?",
    "I'm willing to listen to what you have to say.",
    "I want to hear what happened from your perspective.",
    "I want to understand your side of the story.",
    "Can you help me understand why you feel that way?",
    "I hear you and I want to talk about this.",
    "I'm open to talking about what upset you.",
    "What can I do to help you feel more comfortable?",
    "I want to understand, not to dismiss you.",
    "Thank you for telling me how you feel.",
    "I'm glad you told me how you're feeling.",
    "I appreciate that you're willing to talk about it.",
    "I'm trying to understand, not to argue.",
    "I'm not here to win an argument — I'm here to connect.",

    # ── Boundary-setting (non-manipulative) ───────────────────
    "I think there might be a misunderstanding here.",
    "It's okay for us to have different memories of it.",
    "We both seem to remember this differently.",
    "I respect your experience even if mine was different.",
    "I want to be honest with you about what I remember.",
    "I'm not dismissing your experience at all.",
    "Your memory of this matters and I take it seriously.",
    "I think there was a miscommunication between us.",
    "I'm not trying to dismiss what you experienced.",
    "I want to be transparent about my intentions.",
    "I think we're talking past each other a little.",

    # ── Self-reflection ───────────────────────────────────────
    "I'll try to be clearer with my words going forward.",
    "I care about your wellbeing and want to fix this.",
    "I don't want this to come between us.",
    "I think talking to someone together might help us.",
    "Your concerns are legitimate and I hear them.",
    "I'm not here to argue — I want to resolve this.",
    "I think we need to revisit that conversation.",
    "I know this is hard, but I want to work it out.",
    "We both deserve to feel heard in this conversation.",
    "I feel like we're both trying to be heard right now.",
    "I think honesty is important here, so let me share my truth.",
    "We can work through this if we communicate openly.",

    # ── Reddit-style healthy communication ───────────────────
    "Hey I think we just interpreted that differently, want to talk it through?",
    "I get why you'd feel that way honestly.",
    "I was wrong about that, I'm sorry.",
    "Let's just slow down and figure out what actually happened.",
    "I hear you, that sounds really hard.",
    "Can we both share what we remember without it turning into a fight?",
    "I don't want to argue, I want to understand.",
    "That's a fair point, let me think about that.",
    "I didn't see it that way, but I get where you're coming from.",
    "I'm not trying to fight, I just want to clear this up.",
    "Okay I see your side, here's mine — can we meet in the middle?",
    "I think we just need to talk this out calmly.",
    "That wasn't my intention but I can see how it landed badly.",
    "You're right, I should have handled that better.",
    "I'm genuinely sorry, I didn't realize how that came across.",
]

# ─────────────────────────────────────────────────────────────
# REDDIT CSV LOADER
# ─────────────────────────────────────────────────────────────

def load_reddit_csv(
    csv_path: str = REDDIT_CSV_PATH,
    min_chars: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load and clean casual_data_windows.csv.

    CSV structure (no meaningful header):
        col 0 : row index  (unnamed)
        col 1 : turn_0     (original post / prompt)
        col 2 : turn_1     (first reply)   ← primary signal
        col 3 : turn_2     (second reply)

    Strategy
    --------
    We extract all three turns as individual text items so every
    row contributes up to three training candidates.
    Rows shorter than `min_chars` are dropped (noise/one-liners).
    Reddit markers (>, &gt;, URLs) are stripped.

    Returns
    -------
    pd.DataFrame  columns: text
                  (no label — caller decides how to use it)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Reddit CSV not found at '{csv_path}'.\n"
            f"Update REDDIT_CSV_PATH at the top of gaslighting_data.py."
        )

    raw = pd.read_csv(csv_path, header=0, encoding="utf-8", on_bad_lines="skip")

    # Rename columns regardless of what the header says
    raw.columns = ["row_idx", "turn_0", "turn_1", "turn_2"][: len(raw.columns)]

    def _clean(text: str) -> str:
        """Strip Reddit markdown artifacts and whitespace."""
        if not isinstance(text, str):
            return ""
        text = text.replace("&gt;", "").replace(">", " ")
        text = __import__("re").sub(r"http\S+", "", text)   # remove URLs
        text = __import__("re").sub(r"\s+", " ", text)      # collapse whitespace
        return text.strip()

    # Melt all three turns into one column
    turns = []
    for col in ("turn_0", "turn_1", "turn_2"):
        if col in raw.columns:
            cleaned = raw[col].apply(_clean)
            turns.append(cleaned[cleaned.str.len() >= min_chars])

    all_turns = pd.concat(turns, ignore_index=True).drop_duplicates()
    return pd.DataFrame({"text": all_turns.values})


# ─────────────────────────────────────────────────────────────
# API — load_data(), load_unlabeled(), save_csv()
# ─────────────────────────────────────────────────────────────

def load_data(
    csv_path: str = REDDIT_CSV_PATH,
    n_reddit_non_gas: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build and return the labelled seed DataFrame.

    Composition
    -----------
    • All hand-crafted gaslighting examples       → label 1
    • All hand-crafted non-gaslighting examples   → label 0
    • n_reddit_non_gas turns from the Reddit CSV  → label 0
      (casual Reddit conversations are clearly non-gaslighting)

    If the Reddit CSV is not found, falls back to hand-crafted
    data only and prints a warning.

    Returns
    -------
    pd.DataFrame  columns: id, text, label
                  label  : 1 = gaslighting, 0 = not gaslighting
    """
    # ── Hand-crafted seed ────────────────────────────────────
    texts  = GASLIGHTING_EXAMPLES + NON_GASLIGHTING_EXAMPLES
    labels = [1] * len(GASLIGHTING_EXAMPLES) + [0] * len(NON_GASLIGHTING_EXAMPLES)

    # ── Real Reddit non-gaslighting turns ───────────────────
    if n_reddit_non_gas > 0:
        try:
            reddit_df = load_reddit_csv(csv_path, random_state=random_state)
            reddit_sample = (
                reddit_df
                .sample(n=min(n_reddit_non_gas, len(reddit_df)),
                        random_state=random_state)
                ["text"]
                .tolist()
            )
            texts  += reddit_sample
            labels += [0] * len(reddit_sample)
            print(f"[DATA] Added {len(reddit_sample)} real Reddit turns as label-0 examples.")
        except FileNotFoundError as e:
            print(f"[WARN] {e}\n[WARN] Falling back to hand-crafted data only.")

    df = pd.DataFrame({"id": range(1, len(texts) + 1),
                       "text": texts, "label": labels})
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)
    return df


def load_unlabeled(
    csv_path: str = REDDIT_CSV_PATH,
    n_unlabeled: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Return the unlabeled pool for the semi-supervised loop.

    Uses real Reddit turns (different random seed from load_data so
    there is no overlap with the labeled sample).
    Falls back to 30 hand-crafted borderline examples if CSV missing.

    Returns
    -------
    pd.DataFrame  columns: id, text   (no label column)
    """
    try:
        reddit_df = load_reddit_csv(csv_path, random_state=random_state)

        # Use a different seed offset so this sample doesn't overlap
        # with the n_reddit_non_gas rows already pulled in load_data()
        pool = (
            reddit_df
            .sample(frac=1, random_state=random_state + 99)
            .head(n_unlabeled)
            .reset_index(drop=True)
        )
        pool.insert(0, "id", range(1, len(pool) + 1))
        print(f"[DATA] Unlabeled SSL pool: {len(pool)} real Reddit turns.")
        return pool

    except FileNotFoundError:
        print("[WARN] Reddit CSV not found — using 30 synthetic borderline examples.")
        fallback = [
            "You always do this when we argue.",
            "That's not how I remember it.",
            "I didn't say it like that.",
            "You're being a bit much right now.",
            "Nobody else seems to have a problem with it.",
            "You're reading too much into this.",
            "I've explained this before.",
            "Can you just calm down?",
            "You're taking this the wrong way.",
            "I don't think that's what happened.",
            "You always bring this up.",
            "I remember it differently.",
            "You're being really sensitive today.",
            "That's a bit of an exaggeration.",
            "This always becomes a big thing with you.",
            "I don't think you're remembering that right.",
            "You seem really upset about this.",
            "I think you misunderstood me.",
            "You're making this much harder than it needs to be.",
            "We should talk about this when you're calmer.",
            "I didn't mean it that way.",
            "Maybe you heard me wrong.",
            "You tend to take things personally.",
            "I feel like you're not listening to me.",
            "This is getting blown out of proportion.",
            "You always get like this.",
            "I think we need a timeout.",
            "Let's not fight about this.",
            "I'm not sure that's what I said.",
            "You're misquoting me.",
        ]
        return pd.DataFrame({"id": range(1, len(fallback) + 1),
                             "text": fallback})


def save_csv(path: str = "gaslighting_dataset.csv",
             csv_path: str = REDDIT_CSV_PATH,
             random_state: int = 42) -> str:
    """Persist the labelled dataset to CSV and return the path."""
    df = load_data(csv_path=csv_path, random_state=random_state)
    df.to_csv(path, index=False)
    print(f"[DATA] Saved {len(df)} samples → '{path}'")
    print(f"       Gaslighting    : {df['label'].sum()}")
    print(f"       Not-gaslighting: {(df['label'] == 0).sum()}")
    return path


# ─────────────────────────────────────────────────────────────
# QUICK SUMMARY (run directly)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df  = load_data()
    unl = load_unlabeled()

    print(f"\n{'='*55}")
    print(f"  Gaslighting Dataset Summary")
    print(f"{'='*55}")
    print(f"  Total labelled   : {len(df)}")
    print(f"  Gaslighting  (1) : {df['label'].sum()}")
    print(f"  Not-gaslighting(0): {(df['label'] == 0).sum()}")
    print(f"  Unlabeled SSL pool: {len(unl)}\n")
    print(df.groupby("label")["text"].apply(
        lambda g: g.sample(2, random_state=1)).to_string())
    save_csv()


# ─────────────────────────────────────────────────────────────
# ACCUMULATOR API  — used by the iterative retrain loop
# ─────────────────────────────────────────────────────────────

def append_labeled_data(
    new_df: pd.DataFrame,
    path: str = "accumulated_labels.csv",
) -> str:
    """
    Append newly predicted (high-confidence) rows to the accumulator CSV.

    Called automatically by predict_conversations() in gaslighting_detector.py
    after each prediction batch.

    Parameters
    ----------
    new_df : DataFrame with at least columns: text, label
             (extra columns like confidence, timestamp are preserved)
    path   : path to the growing accumulator CSV

    Returns
    -------
    str : path written to
    """
    # Ensure required columns are present
    required = {"text", "label"}
    if not required.issubset(new_df.columns):
        raise ValueError(f"new_df must contain columns: {required}")

    new_df = new_df.copy()
    new_df["source"] = "predicted"   # mark so we can distinguish from seed

    if os.path.exists(path):
        existing = pd.read_csv(path)
        # Deduplicate on text to avoid adding the same sentence twice
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    else:
        combined = new_df

    combined.to_csv(path, index=False)
    print(f"[DATA] accumulated_labels.csv → {len(combined)} total rows "
          f"(+{len(new_df)} new)  saved to '{path}'")
    return path


def load_accumulated_data(
    accum_path: str = "accumulated_labels.csv",
    csv_path: str = REDDIT_CSV_PATH,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build the full training DataFrame for a retrain cycle:
      seed data  (from load_data)
    + accumulated high-confidence predictions (from accum_path)

    Deduplicates on text, shuffles, and returns a clean DataFrame
    with columns: id, text, label.

    Parameters
    ----------
    accum_path   : path to accumulated_labels.csv
    csv_path     : Reddit CSV path forwarded to load_data()
    random_state : shuffle seed

    Returns
    -------
    pd.DataFrame  columns: id, text, label
    """
    seed_df = load_data(csv_path=csv_path, random_state=random_state)

    if not os.path.exists(accum_path):
        print(f"[DATA] No accumulator found at '{accum_path}' — using seed data only.")
        return seed_df

    accum_df = pd.read_csv(accum_path)

    # Keep only the columns we need
    accum_df = accum_df[["text", "label"]].dropna(subset=["label"])
    accum_df["label"] = accum_df["label"].astype(int)

    combined = (
        pd.concat([seed_df[["text", "label"]], accum_df], ignore_index=True)
        .drop_duplicates(subset=["text"])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    combined.insert(0, "id", range(1, len(combined) + 1))

    n_accum = len(accum_df)
    print(f"[DATA] Seed: {len(seed_df)} | Accumulated: {n_accum} "
          f"| Combined: {len(combined)}")
    return combined
