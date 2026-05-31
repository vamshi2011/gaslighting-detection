#  "That Never Happened"
### Computational Detection of Gaslighting Through Pragmatic and Semantic Linguistic Features

> *"You're imagining things." "That never happened." "You're overreacting."*
> These sentences contain no slurs, no threats — yet they are a recognised form of psychological abuse. This project builds an NLP system that detects them.

---

## Overview

Gaslighting is a form of psychological manipulation in which language is used to systematically undermine another person's memory, perception, or emotional reality. Unlike hate speech or explicit toxicity, it is covert, deniable, and entirely missed by current harmful language detectors.

This project presents a **hybrid NLP classification pipeline** that detects gaslighting in conversational text by combining:
- **Theory-driven pragmatic features** — second-person pronoun density, epistemic verb counts, reality-denial phrase matches (grounded in Sweet, 2019)
- **TF-IDF lexical representations** — unigrams and bigrams
- **SBERT transformer embeddings** — 384-dimensional sentence representations via `all-MiniLM-L6-v2`

Two classifiers are trained and compared: a **Logistic Regression** baseline and an **SBERT + MLP** transformer-based model. A **semi-supervised self-training loop** extends the labelled dataset using high-confidence pseudo-labels from an unlabelled Reddit corpus, and an **iterative retraining architecture** allows the models to improve continuously over deployment cycles.

The full system is deployed as an interactive **Streamlit web application**.

---

## Results

| Model | Accuracy | Precision (gas.) | Recall (gas.) | F1 (gas.) |
|---|---|---|---|---|
| Logistic Regression | 0.767 | 0.842 | 0.696 | 0.762 |
| SBERT + MLP | 0.767 | 0.842 | 0.696 | 0.762 |

Evaluated on a held-out test set of 43 examples (23 gaslighting, 20 not-gaslighting) from a 212-sample manually labelled seed dataset.

---

## Project Structure

```
gaslighting-detection/
│
├── app.py                      # Streamlit UI (Detect / Train / History)
├── gaslighting_detector.py     # Core NLP pipeline
├── gaslighting_data.py         # Dataset loader and accumulator API
├── gaslighting_dataset.csv     # Labelled seed dataset (212 samples)
│
├── requirements.txt
├── README.md
│
├── saved_models/               # Created automatically after training
│   ├── lr_model.pkl            # Trained Logistic Regression
│   ├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectoriser
│   ├── sbert_mlp.pkl           # Trained MLP classifier
│   ├── ling_scaler.pkl         # StandardScaler for linguistic features
│   ├── training_log.json       # Iteration history (metrics + timestamps)
│   ├── confusion_matrices.png  # Generated during training
│   └── top_features.png        # LR feature importance chart
│
└── accumulated_labels.csv      # Grows with each prediction cycle (auto-created)
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/gaslighting-detection.git
cd gaslighting-detection
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download NLTK data** (runs once automatically, or run manually):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## Usage

### Option A — Streamlit Web App (recommended)

```bash
streamlit run app.py
```

Opens in your browser at `http://localhost:8501`. Three pages:

| Page | What it does |
|---|---|
| **Detect** | Classify conversation snippets via free text or CSV/TXT upload. Adjustable confidence threshold. Export results to CSV. Add high-confidence predictions to the training pool. |
| **Train** | Run initial training from scratch or retrain on seed + accumulated labels. Configure SSL threshold and max iterations. View confusion matrix and feature importance plots. |
| **History** | Track model accuracy and F1 across training iterations with an interactive Plotly chart. Export training log as JSON. |

---

### Option B — Command Line

```bash
# Train models from scratch (saved to saved_models/)
python gaslighting_detector.py train

# Classify new text
python gaslighting_detector.py predict "You're overreacting." "Let's talk about this calmly."

# Retrain on seed + all accumulated predictions
python gaslighting_detector.py retrain
```

---

### Option C — Python API

```python
from gaslighting_detector import load_or_train, predict_conversations

# Load saved models (trains from scratch on first run)
models = load_or_train()

# Classify a list of conversation turns
results = predict_conversations(
    texts=["You're imagining things.", "I understand your perspective."],
    models=models,
    confidence_threshold=0.80,
)
print(results[["text", "consensus_label", "consensus_confidence", "is_high_confidence"]])
```

---

## Dataset

`gaslighting_dataset.csv` contains **212 manually labelled examples**:

| Label | Count | Description |
|---|---|---|
| 1 — Gaslighting | 111 | 8 sub-types: reality invalidation, memory attacks, emotional dismissal, sanity attacks, isolation framing, blame-shifting, trust manipulation, perception invalidation |
| 0 — Not Gaslighting | 101 | Healthy disagreement, accountability language, collaborative resolution, active listening |

All examples were authored by the annotator based on language patterns attested in Reddit communities (r/relationship_advice, r/AITA, r/NarcissisticAbuse). No Reddit user data was scraped or stored.

**CSV schema:** `id, text, label`

To use the Reddit unlabelled SSL pool, place `casual_data_windows.csv` in the project root and update `REDDIT_CSV_PATH` at the top of `gaslighting_data.py`.

---

## How It Works

### Feature Engineering

```
Raw Text
   │
   ├── Preprocessing
   │     lowercase → expand contractions → tokenise → remove stopwords → lemmatise
   │     (2nd-person pronouns kept: "you", "your", "yourself")
   │
   ├── Layer 1 — Pragmatic Features (8 dims)
   │     second_person_count   epistemic_verb_count   reality_denial_count
   │     sentiment_polarity    sentiment_subjectivity  polarity_mismatch
   │     word_count            you_density
   │
   ├── Layer 2 — TF-IDF (≤5,000 dims)
   │     unigrams + bigrams, sublinear TF scaling
   │     Layers 1+2 → sparse hstack → Logistic Regression
   │
   └── Layer 3 — SBERT Embeddings (384 dims)
         all-MiniLM-L6-v2, frozen
         Layers 1+3 → dense hstack → MLP Classifier
```

### Semi-Supervised Self-Training Loop

```
Labelled seed  →  Train LR  →  Predict unlabelled pool
                                        │
                              confidence ≥ 0.85?
                               YES ↓         NO → skip
                          Pseudo-label added
                                  ↓
                            Retrain model
                                  ↓
                        Repeat (max 5 iterations)
```

### Iterative Retraining

```
saved_models/    +    accumulated_labels.csv
        │                       │
        └──────────┬────────────┘
                   ↓
          python gaslighting_detector.py retrain
                   ↓
        Updated models + new entry in training_log.json
```

---

## Model Hyperparameters

| Component | Detail |
|---|---|
| **Logistic Regression** | L2 regularisation, C=1.0, lbfgs solver, max_iter=1000, class_weight='balanced' |
| **MLP** | Architecture: 392→128→64→2, ReLU, Adam, lr=0.001, early stopping (patience=15), val_fraction=0.10 |
| **SBERT** | `all-MiniLM-L6-v2` — 384-dim sentence embeddings |
| **Train/test split** | Stratified 80/20, random_state=42 |
| **SSL confidence threshold** | 0.85 default (adjustable in UI and CLI) |

---

## Key References

- Sweet, P. L. (2019). The sociology of gaslighting. *American Sociological Review*, 84(5), 851–875.
- Li, W., et al. (2024). Can a large language model be a gaslighter? *arXiv:2410.09181* (ICLR 2025).
- Wang, Y., et al. (2024). MentalManip: A dataset for fine-grained analysis of mental manipulation. *ACL 2024*.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT. *EMNLP 2019*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.

---

## Limitations

- Test set is small (n=43); results are indicative, not definitive
- Both models produce identical metrics in the current run — SBERT's independent contribution remains unquantified
- All gaslighting examples are author-constructed with no inter-annotator agreement measured
- System operates at utterance level without modelling conversational context

---

## AI Use Disclosure

Claude (Anthropic, 2025) was used as a coding and writing assistant during development. All research design, feature engineering, and result interpretations are the author's own. All citations were independently verified.

---

## Author

**Vamshi Krishna Jinka**  
MSc Computer Science — Constructor University, Bremen  
vjinka@constructor.university
