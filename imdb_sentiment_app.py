# ─────────────────────────────────────────────────────────────────────────────
# AUTO-INSTALL — runs only when a package is missing.
# On Streamlit Cloud this block is skipped because requirements.txt
# already installs everything before the app starts.
# When running locally without a venv it acts as a safety net.
# ─────────────────────────────────────────────────────────────────────────────
import importlib
import subprocess
import sys

REQUIRED_PACKAGES = {
    "nltk":         "nltk>=3.8.1",
    "sklearn":      "scikit-learn>=1.4.0",
    "matplotlib":   "matplotlib>=3.8.0",
    "seaborn":      "seaborn>=0.13.0",
    "pandas":       "pandas>=2.0.0",
    "numpy":        "numpy>=1.26.0",
    "pyarrow":      "pyarrow>=15.0.0",
}

_missing = [
    pip_name
    for import_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]

if _missing:
    import streamlit as st
    with st.spinner(f"Installing missing packages: {', '.join(_missing)} …"):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + _missing
        )
    st.success("Packages installed — reloading app…")
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN IMPORTS (safe to import after the block above)
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── NLTK data downloads ───────────────────────────────────────────────────────
# On Streamlit Cloud these download to a writable temp directory automatically.
for _corpus in ("stopwords", "punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_corpus}" if _corpus.startswith("punkt")
                       else f"corpora/{_corpus}")
    except LookupError:
        nltk.download(_corpus, quiet=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #E50914;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #E50914;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 0.2rem;
    }
    .positive-badge {
        background: #1a7f37;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-size: 1.3rem;
        font-weight: 700;
    }
    .negative-badge {
        background: #c0392b;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-size: 1.3rem;
        font-weight: 700;
    }
    .step-box {
        background: #1e1e2e;
        border-left: 4px solid #E50914;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stButton"] button {
        background-color: #E50914;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #b0070f;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ── NLP helpers ───────────────────────────────────────────────────────────────
ps = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Strip HTML, remove non-alpha chars, lowercase, stem, drop stopwords."""
    text = re.sub(r"<.*?>", " ", str(text))
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = text.split()
    tokens = [ps.stem(w) for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)


@st.cache_data(show_spinner=False)
def preprocess_dataset(df: pd.DataFrame):
    """
    Clean reviews and encode sentiment.
    Columns are cast to plain NumPy types to avoid the PyArrow
    ChunkedArray indexing error in newer pandas versions.
    """
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    df["sentiment_label"] = (
        df["sentiment"].str.lower().str.strip()
        .map({"positive": 1, "negative": 0})
        .astype("int64")
    )
    df = df.dropna(subset=["review", "sentiment_label"]).reset_index(drop=True)

    with st.spinner("🧹 Cleaning & stemming reviews — this may take a moment…"):
        df["clean_review"] = df["review"].astype(str).apply(clean_text)

    return df


def build_vectorizer(choice: str, max_features: int):
    if choice == "CountVectorizer":
        return CountVectorizer(max_features=max_features)
    return TfidfVectorizer(max_features=max_features)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Reds",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")
    ax.set_title("Confusion Matrix", color="white")
    ax.tick_params(colors="white")
    plt.tight_layout()
    return fig


def plot_sentiment_dist(df: pd.DataFrame):
    counts = df["sentiment"].str.lower().str.strip().value_counts()
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    bars = ax.bar(
        counts.index, counts.values,
        color=["#1a7f37", "#c0392b"],
        edgecolor="none", width=0.5,
    )
    ax.set_title("Sentiment Distribution", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + counts.max() * 0.01,
            f"{val:,}", ha="center", color="white", fontsize=9,
        )
    plt.tight_layout()
    return fig


# ── Session state ─────────────────────────────────────────────────────────────
for _key in ("model_trained", "classifier", "vectorizer", "accuracy",
             "report_df", "y_test", "y_pred", "processed_df"):
    if _key not in st.session_state:
        st.session_state[_key] = None

st.session_state.setdefault("model_trained", False)

# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">🎬 IMDB Sentiment Analyzer</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Upload your IMDB dataset · Choose a vectorizer · Train · Predict'
    '</div>',
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("1 · Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload IMDB CSV (review, sentiment)",
        type=["csv"],
        help="CSV must have 'review' and 'sentiment' columns.",
    )

    st.subheader("2 · Data Options")
    use_sample = st.checkbox("Use a subset of rows (faster training)", value=True)
    sample_size = st.slider(
        "Max rows to use", 1000, 50000, 10000, step=1000,
        disabled=not use_sample,
    )

    st.subheader("3 · Train / Test Split")
    test_size    = st.slider("Test set size (%)", 10, 40, 20, step=5)
    random_state = st.number_input(
        "Random seed", min_value=0, max_value=999, value=42, step=1
    )

    st.subheader("4 · Vectorizer")
    vec_choice = st.radio(
        "Choose vectorizer",
        ["CountVectorizer", "TF-IDF"],
        help="CountVectorizer uses raw word counts; TF-IDF weights by rarity.",
    )
    max_features = st.slider(
        "Max features (vocabulary size)", 1000, 50000, 10000, step=1000,
    )

    st.subheader("5 · Model")
    alpha = st.slider("BernoulliNB alpha (smoothing)", 0.1, 2.0, 1.0, step=0.1)

    train_button = st.button("🚀 Train Model", use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_train, tab_predict, tab_guide = st.tabs(
    ["📊 Train & Evaluate", "🔮 Predict Sentiment", "📖 How It Works"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Train & Evaluate
# ─────────────────────────────────────────────────────────────────────────────
with tab_train:
    if not uploaded_file:
        st.info("⬅️ Upload your IMDB CSV file in the sidebar to get started.")
        st.stop()

    # Load CSV — dtype=str prevents pandas choosing Arrow-backed dtypes
    try:
        df_raw = pd.read_csv(uploaded_file, dtype=str)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        st.stop()

    df_raw.columns = [c.lower().strip() for c in df_raw.columns]

    if "review" not in df_raw.columns or "sentiment" not in df_raw.columns:
        st.error("CSV must contain **'review'** and **'sentiment'** columns.")
        st.stop()

    # Preview
    st.subheader("📋 Dataset Preview")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.dataframe(df_raw[["review", "sentiment"]].head(5), use_container_width=True)
    with col_b:
        st.pyplot(plot_sentiment_dist(df_raw))

    # Optional subset
    if use_sample and len(df_raw) > sample_size:
        df_raw = df_raw.sample(n=sample_size, random_state=42).reset_index(drop=True)
        st.caption(f"Using a random sample of **{sample_size:,}** rows.")

    # ── Train ──────────────────────────────────────────────────────────────
    if train_button:
        st.session_state["model_trained"] = False

        df_clean = preprocess_dataset(df_raw)
        st.session_state["processed_df"] = df_clean

        # Convert to plain NumPy arrays — fixes PyArrow ChunkedArray error
        X = np.array(df_clean["clean_review"].tolist())
        y = np.array(df_clean["sentiment_label"].tolist(), dtype=np.int64)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size / 100,
            random_state=int(random_state),
            stratify=y,
        )

        with st.spinner(f"Vectorizing with {vec_choice}…"):
            vectorizer  = build_vectorizer(vec_choice, max_features)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec  = vectorizer.transform(X_test)

        with st.spinner("Training BernoulliNB…"):
            clf = BernoulliNB(alpha=alpha)
            clf.fit(X_train_vec, y_train)

        y_pred    = clf.predict(X_test_vec)
        acc       = accuracy_score(y_test, y_pred)
        report    = classification_report(
            y_test, y_pred,
            target_names=["Negative", "Positive"],
            output_dict=True,
        )
        report_df = pd.DataFrame(report).T.round(3)

        st.session_state.update({
            "model_trained": True,
            "classifier":    clf,
            "vectorizer":    vectorizer,
            "accuracy":      acc,
            "report_df":     report_df,
            "y_test":        y_test,
            "y_pred":        y_pred,
        })
        st.success("✅ Model trained successfully!")

    # ── Results ────────────────────────────────────────────────────────────
    if st.session_state["model_trained"]:
        acc       = st.session_state["accuracy"]
        report_df = st.session_state["report_df"]
        y_test    = st.session_state["y_test"]
        y_pred    = st.session_state["y_pred"]

        st.markdown("---")
        st.subheader("📈 Model Performance")

        m1, m2, m3, m4 = st.columns(4)
        neg_f1  = report_df.loc["Negative", "f1-score"]
        pos_f1  = report_df.loc["Positive", "f1-score"]
        support = (
            int(report_df.loc["Negative", "support"])
            + int(report_df.loc["Positive", "support"])
        )

        for col, label, value in [
            (m1, "Accuracy",      f"{acc * 100:.2f}%"),
            (m2, "F1 – Negative", f"{neg_f1:.3f}"),
            (m3, "F1 – Positive", f"{pos_f1:.3f}"),
            (m4, "Test Samples",  f"{support:,}"),
        ]:
            col.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Classification Report**")
            st.dataframe(
                report_df.style.format("{:.3f}", na_rep="-"),
                use_container_width=True,
            )
        with col_right:
            st.markdown("**Confusion Matrix**")
            st.pyplot(plot_confusion_matrix(y_test, y_pred))

        with st.expander("ℹ️ Training Configuration Summary"):
            train_n = int(len(y_test) / (test_size / 100) * (1 - test_size / 100))
            st.json({
                "vectorizer":    vec_choice,
                "max_features":  max_features,
                "test_size_pct": test_size,
                "random_state":  int(random_state),
                "nb_alpha":      alpha,
                "train_samples": train_n,
                "test_samples":  int(len(y_test)),
            })

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Predict
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.subheader("🔮 Predict Sentiment of a New Review")

    if not st.session_state["model_trained"]:
        st.warning("⚠️ Please upload a dataset and train the model first (Tab 1).")
    else:
        st.markdown(
            "Enter any movie review below. The trained model will classify "
            "it as **Positive** or **Negative**."
        )

        user_review = st.text_area(
            "Movie Review",
            height=160,
            placeholder="e.g. This movie was absolutely brilliant! The acting was superb…",
        )

        if st.button("🔍 Predict", key="predict_btn"):
            if not user_review.strip():
                st.warning("Please enter a review before predicting.")
            else:
                cleaned    = clean_text(user_review)
                vec        = st.session_state["vectorizer"]
                clf        = st.session_state["classifier"]
                X_input    = vec.transform([cleaned])
                prediction = clf.predict(X_input)[0]
                proba      = clf.predict_proba(X_input)[0]

                st.markdown("---")
                st.markdown("### Result")

                col_res, col_prob = st.columns([1, 2])
                with col_res:
                    badge_cls  = "positive-badge" if prediction == 1 else "negative-badge"
                    badge_text = "😊 POSITIVE"    if prediction == 1 else "😞 NEGATIVE"
                    st.markdown(
                        f'<div style="text-align:center;padding:2rem 0">'
                        f'<span class="{badge_cls}">{badge_text}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                with col_prob:
                    st.markdown("**Prediction Confidence**")
                    fig_p, ax_p = plt.subplots(figsize=(4, 1.8))
                    fig_p.patch.set_facecolor("#1e1e2e")
                    ax_p.set_facecolor("#1e1e2e")
                    bars = ax_p.barh(
                        ["Negative", "Positive"],
                        [proba[0], proba[1]],
                        color=["#c0392b", "#1a7f37"],
                        edgecolor="none",
                    )
                    for bar, val in zip(bars, [proba[0], proba[1]]):
                        ax_p.text(
                            val + 0.01, bar.get_y() + bar.get_height() / 2,
                            f"{val * 100:.1f}%", va="center",
                            color="white", fontsize=9,
                        )
                    ax_p.set_xlim(0, 1.15)
                    ax_p.tick_params(colors="white")
                    ax_p.spines[:].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig_p)

                with st.expander("🔬 Cleaned Review (after preprocessing)"):
                    st.write(cleaned if cleaned else "(empty after cleaning)")

        # Batch predict
        st.markdown("---")
        st.subheader("📂 Batch Predict from CSV")
        st.markdown(
            "Upload a CSV with a **`review`** column. "
            "The app will add a **`predicted_sentiment`** column you can download."
        )
        batch_file = st.file_uploader(
            "Upload batch CSV", type=["csv"], key="batch_uploader"
        )
        if batch_file:
            batch_df = pd.read_csv(batch_file, dtype=str)
            batch_df.columns = [c.lower().strip() for c in batch_df.columns]
            if "review" not in batch_df.columns:
                st.error("Batch CSV must have a 'review' column.")
            else:
                with st.spinner("Running batch predictions…"):
                    clean_reviews = [
                        clean_text(r) for r in batch_df["review"].tolist()
                    ]
                    X_batch = st.session_state["vectorizer"].transform(clean_reviews)
                    preds   = st.session_state["classifier"].predict(X_batch)
                    batch_df["predicted_sentiment"] = np.where(
                        preds == 1, "positive", "negative"
                    )
                st.success(f"Predicted {len(batch_df):,} reviews!")
                st.dataframe(
                    batch_df[["review", "predicted_sentiment"]].head(20),
                    use_container_width=True,
                )
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=batch_df[["review", "predicted_sentiment"]].to_csv(index=False),
                    file_name="imdb_predictions.csv",
                    mime="text/csv",
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — How It Works
# ─────────────────────────────────────────────────────────────────────────────
with tab_guide:
    st.subheader("📖 How This App Works")

    steps = [
        ("1 · Upload Dataset",
         "Upload a CSV with <b>review</b> (text) and <b>sentiment</b> "
         "(positive/negative) columns. The IMDB 50K dataset from Kaggle works out of the box."),
        ("2 · Data Preprocessing",
         "HTML tags are stripped, non-alphabetic characters removed, text lowercased, "
         "English stopwords dropped, and each word is reduced to its root via <b>Porter Stemming</b>."),
        ("3 · Train / Test Split",
         "Data is split into training and test sets (control the ratio via the sidebar). "
         "Stratification ensures balanced class proportions in both sets."),
        ("4 · Vectorization",
         "<b>CountVectorizer</b> converts each review into a word-count vector. "
         "<b>TF-IDF</b> additionally down-weights common words and up-weights rare informative ones."),
        ("5 · BernoulliNB Classifier",
         "A Bernoulli Naïve Bayes model is trained on the vectorized data. "
         "It works excellently for text sentiment classification."),
        ("6 · Evaluation",
         "Accuracy, precision, recall, F1-score per class, and the confusion matrix are shown "
         "on the unseen test set."),
        ("7 · Predict",
         "Type any review in the <b>Predict</b> tab to get <b>Positive / Negative</b> "
         "with confidence probabilities. Batch CSV prediction is also supported."),
    ]

    for title, body in steps:
        st.markdown(
            f'<div class="step-box"><strong>{title}</strong><br>{body}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Tips for better accuracy**")
    st.markdown(
        "- Use the full 50K dataset (uncheck *Use a subset*).  \n"
        "- Try TF-IDF with 20K–30K features.  \n"
        "- Keep test split at 20–30%.  \n"
        "- BernoulliNB `alpha=1.0` (Laplace smoothing) is a solid default."
    )

    st.markdown("---")
    st.markdown(
        "**Deploying to Streamlit Cloud?**  \n"
        "Push both `imdb_sentiment_app.py` and `requirements.txt` to your GitHub repo, "
        "then connect the repo at [share.streamlit.io](https://share.streamlit.io). "
        "Streamlit Cloud will install all packages from `requirements.txt` automatically."
    )
    st.markdown(
        "**Run locally:**  \n"
        "```\npip install -r requirements.txt\nstreamlit run imdb_sentiment_app.py\n```"
    )
