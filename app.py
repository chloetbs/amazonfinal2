"""
=======================================================================
Amazon Product Recommendation System — TBS Barcelona, Group 7
=======================================================================

Three recommendation strategies:
  1. Popularity-Based   → Cold Start fallback for new / unknown users
  2. User-Based CF      → "Users who shop like you also bought…"
  3. Item-Based CF      → "Customers who bought X also bought Y…"

Hybrid Effective Rating (Lecture 3):
    EffectiveRating = Rating + 0.1 × Sentiment

Cosine Similarity (Lecture 4):
    sim(X, Y) = (X · Y) / (||X|| × ||Y||)
=======================================================================
"""

import os
import re
import time
import string
import random
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import streamlit as st

# Plotly for interactive EDA charts — install with: pip install plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATA_FILE = "Group7.xlsx"

TEXT_COL_CANDIDATES = [
    "ReviewText", "Text", "Review", "Reviews",
    "Summary", "review_text", "text",
]

# Significance weighting threshold:
# Products must share at least this many common raters
# for their similarity to be fully trusted.
SIGNIFICANCE_THRESHOLD = 10 


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UTILITY FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_text_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the review text column (case-insensitive match)."""
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in TEXT_COL_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


@st.cache_resource(show_spinner=False)
def load_sentiment_lexicons() -> Tuple[set, set]:
    """
    Load Hu & Liu opinion lexicon (~2 000 positive, ~4 800 negative words).
    Falls back to a small built-in set if files are missing. 
    """
    def _load(path, fallback):
        if not os.path.exists(path):
            return fallback
        words = set()
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(";") and not line.startswith("#"):
                    words.add(line.lower())
        return words if words else fallback

    pos = _load(
        os.path.join(os.getcwd(), "positive-words.txt"),
        {"good", "great", "excellent", "amazing", "fantastic",
         "love", "perfect", "satisfied", "happy", "recommend", "awesome"},
    )
    neg = _load(
        os.path.join(os.getcwd(), "negative-words.txt"),
        {"bad", "terrible", "awful", "poor", "hate", "disappointed",
         "worst", "broken", "refund", "return", "useless"},
    )
    return pos, neg


def sentiment_score(text: str) -> float:
    """
    Lexicon-based sentiment:  (positive_count - negative_count) / token_count
    Returns value in [-1, 1].
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    pos_words, neg_words = load_sentiment_lexicons()
    tokens = [t.strip(".,!?;:()[]\"'").lower() for t in text.split()]
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in pos_words)
    neg = sum(1 for t in tokens if t in neg_words)
    return float(np.clip((pos - neg) / len(tokens), -1.0, 1.0))


def compute_effective_rating(row: pd.Series, text_col: Optional[str]) -> float:
    """
    Hybrid Effective Rating (Lecture 3):
        EffectiveRating = Rating + 0.1 * Sentiment

    The 0.1 weight ensures sentiment adjusts the rating gently.
    A 5-star review with negative text becomes ~4.9, not 3.0.
    """
    rating = row["Rating"]
    if pd.isna(rating):
        return np.nan
    if text_col is None:
        return float(rating)
    return float(rating + 0.1 * sentiment_score(row[text_col]))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING & CLEANING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data(show_spinner="Loading and cleaning data...")
def load_and_clean_data(data_path: str, min_ratings: int = 5):

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"'{data_path}' not found. Place it next to app.py.")

    df = pd.read_excel(data_path)
    for col in ("UserId", "ProductId"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    # ── DATA CLEANING ──────────────────────────────
    df["UserId"] = df["UserId"].str.replace(r'(_\d+)+$', '', regex=True)
    if "Reviews" in df.columns:
        df["Reviews"] = df["Reviews"].fillna("").str.strip()
    text_col_name = find_text_column(df)
    dedup_cols = ["UserId", "ProductId", "Rating"]
    if text_col_name:
        dedup_cols.append(text_col_name)
    df = df.drop_duplicates(subset=dedup_cols)
    # ──────────────────────────────────────────────

    required = {"UserId", "ProductId", "Rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Rating"])

    df_raw = df.copy()
    all_user_ids = df["UserId"].unique().tolist()

    id_to_name = {}
    if "product_name" in df.columns:
        id_to_name = (
            df[["ProductId", "product_name"]]
            .dropna(subset=["ProductId"])
            .drop_duplicates("ProductId")
            .set_index("ProductId")["product_name"]
            .astype(str).to_dict()
        )

    text_col = find_text_column(df)

    counts = df.groupby("ProductId")["Rating"].count()
    valid = counts[counts >= min_ratings].index
    df_clean = df[df["ProductId"].isin(valid)].copy()

    df_clean["EffectiveRating"] = df_clean.apply(
        lambda r: compute_effective_rating(r, text_col), axis=1
    )
    df_clean = df_clean.dropna(subset=["EffectiveRating"])

    if text_col:
        df_clean["has_text_review"] = df_clean[text_col].apply(
            lambda v: isinstance(v, str) and len(str(v).strip()) >= 5
        )
    else:
        df_clean["has_text_review"] = False
    df_clean["sentiment_component"] = df_clean["EffectiveRating"] - df_clean["Rating"]

    user_item = df_clean.pivot_table(
        index="UserId", columns="ProductId",
        values="EffectiveRating", fill_value=0.0,
    )

    sim_counts = df_clean.groupby("ProductId")["Rating"].count()
    valid_sim = sim_counts[sim_counts >= min_ratings].index
    item_vectors = user_item.T.loc[valid_sim]

    raw_sim = cosine_similarity(item_vectors)
    binary_matrix = (item_vectors.values > 0).astype(float)
    common_raters = binary_matrix @ binary_matrix.T
    sig_weights = np.minimum(common_raters, SIGNIFICANCE_THRESHOLD) / SIGNIFICANCE_THRESHOLD
    weighted_sim = raw_sim * sig_weights

    item_similarity = pd.DataFrame(
        weighted_sim, index=item_vectors.index, columns=item_vectors.index
    )

    return df_raw, df_clean, user_item, item_similarity, text_col, all_user_ids, id_to_name

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ADMIN HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AMAZON_ID_RE = re.compile(r'^[A-Z0-9]{28}$')

def is_valid_amazon_id(uid: str) -> bool:
    """28 chars, starts with A, uppercase letters and digits only."""
    return bool(AMAZON_ID_RE.match(uid)) and uid.startswith("A")

def generate_amazon_id() -> str:
    """Generate a random Amazon-style user ID."""
    chars = string.ascii_uppercase + string.digits
    return "A" + "".join(random.choices(chars, k=27))

def get_master_df() -> pd.DataFrame:
    """Returns the cleaned master DataFrame from session state."""
    return st.session_state["master_df"]

def append_row_to_master(new_row: dict) -> None:
    df = get_master_df()
    st.session_state["master_df"] = pd.concat(
        [df, pd.DataFrame([new_row])], ignore_index=True
    )
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(show_spinner="Evaluating model...")
def evaluate_model(df_clean: pd.DataFrame, min_ratings: int = 5) -> dict:
    """
    80/20 train/test evaluation of Item-Based CF.

    Metrics:
      RMSE  - root mean square error of predicted vs actual ratings
      MAE   - mean absolute error
      Precision@5 - when we predict >= 4 stars, how often is actual >= 4 stars?
    """
    uc = df_clean.groupby("UserId").size()
    eligible = uc[uc >= 5].index
    df_e = df_clean[df_clean["UserId"].isin(eligible)].copy()

    if len(df_e) < 100:
        return {"error": "Not enough data for evaluation."}

    train_parts, test_parts = [], []
    for _, grp in df_e.groupby("UserId"):
        if len(grp) >= 2:
            tr, te = train_test_split(grp, test_size=0.2, random_state=42)
            train_parts.append(tr)
            test_parts.append(te)

    if not train_parts:
        return {"error": "Not enough users with multiple ratings."}

    train = pd.concat(train_parts)
    test = pd.concat(test_parts)

    pivot = train.pivot_table(
        index="UserId", columns="ProductId",
        values="EffectiveRating", fill_value=0.0,
    )
    vecs = pivot.T
    sim = cosine_similarity(vecs)
    sim_df = pd.DataFrame(sim, index=vecs.index, columns=vecs.index)

    preds, actuals = [], []
    for _, row in test.iterrows():
        u, p, actual = row["UserId"], row["ProductId"], row["EffectiveRating"]
        if u not in pivot.index or p not in sim_df.index:
            continue
        user_ratings = pivot.loc[u]
        rated = user_ratings[user_ratings > 0].index
        rated_in_sim = [x for x in rated if x in sim_df.index and x != p]
        if not rated_in_sim:
            continue
        s = sim_df.loc[p, rated_in_sim]
        r = user_ratings[rated_in_sim]
        denom = s.abs().sum()
        if denom == 0:
            continue
        preds.append(float((s * r).sum() / denom))
        actuals.append(float(actual))

    if not preds:
        return {"error": "Could not generate enough predictions."}

    p_arr, a_arr = np.array(preds), np.array(actuals)
    rmse = float(np.sqrt(np.mean((p_arr - a_arr) ** 2)))
    mae = float(np.mean(np.abs(p_arr - a_arr)))

    high = p_arr >= 4.0
    prec = float((a_arr[high] >= 4.0).mean()) if high.sum() > 0 else 0.0

    return {
        "rmse": rmse, "mae": mae, "precision": prec,
        "n_preds": len(preds), "n_train": len(train), "n_test": len(test),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RECOMMENDATION FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_similar_products(pid: str, sim_df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """Return top_n most similar products (excluding self) from the similarity matrix."""
    if pid not in sim_df.index:
        raise KeyError(f"Product '{pid}' not in similarity matrix.")
    scores = sim_df.loc[pid].drop(labels=[pid])
    return scores.sort_values(ascending=False).head(top_n)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STREAMLIT APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    st.set_page_config(
        page_title="Amazon Recommender - TBS Group 7",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CSS - Complete visual overhaul
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Base */
    .stApp {
        background: linear-gradient(180deg, #F7F8FA 0%, #EEF0F3 100%);
        font-family: 'Inter', -apple-system, sans-serif;
        color: #1a1a2e;
    }

    /* Hero Banner */
    .hero {
        background: linear-gradient(135deg, #131921 0%, #232F3E 50%, #37475A 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(255,153,0,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero h1 {
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.02em;
    }
    .hero h1 span { color: #FF9900; }
    .hero p {
        color: rgba(255,255,255,0.8);
        font-size: 0.95rem;
        margin: 0;
        font-weight: 400;
    }

    /* Stat Cards */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E8E8E8;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    .stat-card .number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #232F3E;
        line-height: 1.2;
    }
    .stat-card .number.orange { color: #FF9900; }
    .stat-card .label {
        font-size: 0.78rem;
        color: #666;
        font-weight: 500;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Product Cards */
    .product-card {
        background: #FFFFFF;
        border: 1px solid #E8E8E8;
        border-radius: 10px;
        padding: 1.1rem;
        height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
        position: relative;
    }
    .product-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    .product-name {
        font-weight: 700;
        font-size: 0.88rem;
        color: #0F1111;
        margin-bottom: 0.35rem;
        line-height: 1.3;
        min-height: 2.2rem;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .product-stars { font-size: 0.85rem; margin-bottom: 0.25rem; }
    .product-meta {
        font-size: 0.75rem;
        color: #565959;
        margin-bottom: 0.4rem;
        line-height: 1.4;
    }

    /* Tags */
    .tag {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 100px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .tag-green  { background: #E8F5E9; color: #2E7D32; }
    .tag-red    { background: #FFEBEE; color: #C62828; }
    .tag-gray   { background: #F5F5F5; color: #616161; }

    .match-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-top: 0.35rem;
    }
    .match-high   { background: #FFF3E0; color: #E65100; border: 1px solid #FFB74D; }
    .match-medium { background: #FFF8E1; color: #F57F17; border: 1px solid #FFD54F; }
    .match-low    { background: #F5F5F5; color: #757575; border: 1px solid #E0E0E0; }
    .match-trend  { background: #E3F2FD; color: #1565C0; border: 1px solid #90CAF9; }

    /* Section Headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #232F3E;
        margin: 0.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #FF9900;
        display: inline-block;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(180deg, #FFB84D 0%, #FF9900 100%);
        color: #111;
        border: 1px solid #E68A00;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(180deg, #FFC966 0%, #FFB84D 100%);
        box-shadow: 0 2px 8px rgba(255,153,0,0.3);
        transform: translateY(-1px);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #FFFFFF;
        border-radius: 10px;
        padding: 0.3rem;
        border: 1px solid #E8E8E8;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.5rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background: #232F3E !important;
        color: #FFFFFF !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #232F3E;
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    /* Footer */
    .footer {
        background: #232F3E;
        color: rgba(255,255,255,0.7);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
        font-size: 0.8rem;
        text-align: center;
    }

    /* Insight callout box */
    .insight-box {
        background: #FFFBF0;
        border-left: 4px solid #FF9900;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #333;
    }

    /* Admin review rows  (new) */
    .review-row {
        background: #FAFAFA;
        border-left: 3px solid #FF9900;
        border-radius: 0 6px 6px 0;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.55rem;
        font-size: 0.85rem;
    }
    .review-row-product {
        font-weight: 700;
        font-size: 0.9rem;
        color: #232F3E;
    }
    .review-row-id {
        font-size: 0.72rem;
        color: #888;
        font-family: monospace;
    }
    .review-row-meta {
        font-size: 0.78rem;
        color: #777;
        margin: 0.15rem 0 0.2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HERO
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("""
    <div class="hero">
        <h1>🛒 Amazon <span>Recommendation</span> Engine</h1>
        <p>TBS Barcelona - Group 7 &nbsp;|&nbsp; Collaborative Filtering &nbsp;|&nbsp;
        Hybrid Rating (Star + Sentiment) &nbsp;|&nbsp; Cosine Similarity</p>
    </div>
    """, unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SIDEBAR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with st.sidebar:
        st.markdown("### Settings")
        min_ratings = st.slider(
            "Cold Start Filter (min ratings per product)",
            1, 25, 5,
            help="Products with fewer ratings are excluded to reduce noise.",
        )
        num_recs = st.slider("Recommendations to show", 5, 30, 10)

        st.divider()
        st.markdown("### How It Works")
        st.markdown("""
        **1. Popularity** - trending products for new users

        **2. User-Based CF** - finds users with similar taste

        **3. Item-Based CF** - finds similar products

        **Hybrid Rating** = Stars + 0.1 x Sentiment

        **Match Score** = Cosine Similarity x Significance Weight
        """)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LOAD DATA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    t0 = time.time()
    try:
        (df_raw, df_clean, user_item, item_similarity,
         text_col, all_user_ids, id_to_name) = load_and_clean_data(DATA_FILE, min_ratings)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()
    load_time = time.time() - t0

    if "master_df" not in st.session_state:
        st.session_state["master_df"] = df_raw

    product_ids = user_item.columns.astype(str).tolist()
    all_uids = set(str(u) for u in all_user_ids)
    user_lookup = {str(u): u for u in user_item.index}
    top_n = num_recs

    if not product_ids:
        st.warning("No products left after filtering. Lower the Cold Start threshold.")
        st.stop()

    # Product labels: "Name (ID)" for searchable dropdown
    product_options = []
    label_to_id = {}
    for pid in product_ids:
        name = id_to_name.get(pid, "")
        label = f"{name[:55]} ({pid})" if name and name != pid else pid
        product_options.append(label)
        label_to_id[label] = pid

    # Summary stats per product
    summary = (
        df_clean.groupby("ProductId").agg(
            n_ratings=("Rating", "count"),
            avg_rating=("Rating", "mean"),
            avg_effective=("EffectiveRating", "mean"),
            has_text=("has_text_review", "max"),
            avg_sent=("sentiment_component", "mean"),
        ).reset_index()
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RECOMMENDATION FUNCTIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def popularity_recs(k=5):
        r = summary[summary["n_ratings"] >= 5].nlargest(k, "avg_effective").copy()
        r["similarity"] = np.nan
        return r

    def user_based_recs(uid_str, k=5):
        if uid_str not in user_lookup:
            return None
        target = user_lookup[uid_str]
        if target not in user_item.index:
            return None

        vec = user_item.loc[target].values.reshape(1, -1)
        sims = cosine_similarity(vec, user_item.values).flatten()
        sim_s = pd.Series(sims, index=user_item.index)
        sim_s.loc[target] = 0.0

        mat = user_item.copy()
        rated_mask = mat.loc[target] > 0
        num = (mat.T * sim_s).T.sum(axis=0)
        den = ((mat > 0).T * np.abs(sim_s)).T.sum(axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            pred = num / den
        pred = pred.replace([np.inf, -np.inf], np.nan)
        pred = pred[~rated_mask].dropna()

        if pred.empty:
            return None

        df_pred = (
            pred.sort_values(ascending=False).head(k)
            .rename("predicted").to_frame()
            .reset_index().rename(columns={"index": "ProductId"})
        )
        df_pred = df_pred.merge(summary, on="ProductId", how="left")
        df_pred["similarity"] = (df_pred["predicted"] - 1) / 4.0
        return df_pred

    def item_based_recs(pid, k=5):
        try:
            similar = get_similar_products(pid, item_similarity, top_n=k)
        except Exception:
            return None
        df_r = (
            similar.rename("similarity").reset_index()
            .rename(columns={"index": "ProductId"})
        )
        df_r = df_r.merge(summary, on="ProductId", how="left")
        df_r = df_r.sort_values(
            ["avg_effective", "similarity", "n_ratings"],
            ascending=[False, False, False],
        )
        return df_r

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CARD RENDERER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def stars_html(val):
        if pd.isna(val):
            return "---"
        v = float(np.clip(val, 0, 5))
        full = int(v)
        half = 1 if (v - full) >= 0.25 and (v - full) < 0.75 else 0
        return "⭐" * full + ("½" if half else "") + "☆" * max(0, 5 - full - half)

    def render_cards(df):
        if df is None or df.empty:
            st.info("No recommendations found. Try different inputs.")
            return

        df = df.copy()
        df["display_name"] = df["ProductId"].map(id_to_name).fillna(df["ProductId"])
        df = df.sort_values(
            ["avg_effective", "similarity", "n_ratings"],
            ascending=[False, False, False],
        ).head(top_n)

        for start in range(0, len(df), 5):
            chunk = df.iloc[start:start+5]
            cols = st.columns(min(5, len(chunk)))
            for col, (_, row) in zip(cols, chunk.iterrows()):
                with col:
                    name = str(row["display_name"])
                    if len(name) > 50:
                        name = name[:48] + "..."

                    sim = row.get("similarity", np.nan)
                    n_r = int(row["n_ratings"]) if not pd.isna(row.get("n_ratings")) else 0
                    avg_e = row.get("avg_effective", np.nan)

                    # Match badge
                    if pd.isna(sim):
                        badge = f'<span class="match-badge match-trend">⭐ Top Rated</span>'
                    else:
                        pct = float(sim) * 100 if 0 <= float(sim) <= 1 else float(np.clip(sim / 5 * 100, 0, 100))
                        if pct >= 70:
                            cls = "match-high"
                        elif pct >= 40:
                            cls = "match-medium"
                        else:
                            cls = "match-low"
                        badge = f'<span class="match-badge {cls}">Match: {pct:.1f}%</span>'

                    # Sentiment tag
                    has_txt = bool(row.get("has_text", False))
                    sent = float(row.get("avg_sent", 0))
                    if not has_txt:
                        tag = '<span class="tag tag-gray">No reviews</span>'
                    elif sent > 0.005:
                        tag = '<span class="tag tag-green">Positive</span>'
                    elif sent < -0.005:
                        tag = '<span class="tag tag-red">Negative</span>'
                    else:
                        tag = '<span class="tag tag-gray">Neutral</span>'

                    s = stars_html(avg_e if not pd.isna(avg_e) else row.get("avg_rating"))

                    st.markdown(f"""
                    <div class="product-card">
                        <div class="product-name">{name}</div>
                        <div class="product-stars">{s}</div>
                        <div class="product-meta">{n_r} ratings</div>
                        {tag}<br/>
                        {badge}
                    </div>
                    """, unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TABS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👤 User-Based", "📦 Product-Based",
        "📊 Data Insights", "🧪 Evaluation", "🛠️ Admin",
    ])

    # -- TAB 1: USER-BASED --
    with tab1:
        st.markdown('<div class="section-header">Personalized Recommendations</div>', unsafe_allow_html=True)
        st.markdown("Enter your **User ID** to discover products matched to your taste. Unknown users get trending best sellers.")

        c1, c2 = st.columns([3, 1])
        with c1:
            uid_input = st.text_input(
                "User ID", placeholder="e.g. AFA6RPDU",
                help="Copy a UserId from the first column of Group7.xlsx",
            )
        with c2:
            st.markdown("<br/>", unsafe_allow_html=True)
            go_user = st.button("Recommend", key="btn_user", use_container_width=True)

        if go_user:
            # Also recognise users added via Admin this session
            live_uids = set(get_master_df()["UserId"].astype(str).unique())
            uid = uid_input.strip()
            if not uid:
                st.warning("Please enter a User ID.")
            elif uid in live_uids:
                recs = user_based_recs(uid, k=top_n)
                if recs is not None and not recs.empty:
                    st.markdown(f'<div class="insight-box"><b>Found {len(recs)} recommendations</b> for user <code>{uid}</code> based on {user_item.shape[0]:,} similar shoppers.</div>', unsafe_allow_html=True)
                    render_cards(recs)
                else:
                    st.info("Your taste is unique! Here are our best sellers instead.")
                    render_cards(popularity_recs(top_n))
            else:
                st.info("New user detected - showing trending best sellers.")
                render_cards(popularity_recs(top_n))

    # -- TAB 2: PRODUCT-BASED --
    with tab2:
        st.markdown('<div class="section-header">Similar Products</div>', unsafe_allow_html=True)
        st.markdown("Select a product to find items that similar customers also purchased.")

        c1, c2 = st.columns([3, 1])
        with c1:
            selected_label = st.selectbox(
                "Search by product name",
                options=product_options,
                index=0,
            )
        with c2:
            st.markdown("<br/>", unsafe_allow_html=True)
            go_item = st.button("Find Similar", key="btn_item", use_container_width=True)

        sel_pid = label_to_id[selected_label]

        if go_item:
            recs = item_based_recs(sel_pid, k=top_n)
            if recs is not None and not recs.empty:
                target_name = id_to_name.get(sel_pid, sel_pid)
                st.markdown(f'<div class="insight-box">Showing products similar to <b>{target_name[:60]}</b>. Match scores use <b>significance-weighted cosine similarity</b> - products with few common raters are penalized.</div>', unsafe_allow_html=True)
                render_cards(recs)
            else:
                st.info("No similar products found for this selection.")

    # -- TAB 3: DATA INSIGHTS --
    with tab3:
        st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
        st.markdown("Understanding the dataset before building any model. These metrics reveal **sparsity**, **distribution**, and **sentiment** patterns.")

        n_users = df_raw["UserId"].nunique()
        n_prods = df_raw["ProductId"].nunique()
        n_ratings = len(df_raw)
        n_users_c = df_clean["UserId"].nunique()
        n_prods_c = df_clean["ProductId"].nunique()
        n_ratings_c = len(df_clean)
        total_cells = n_users_c * n_prods_c
        sparsity = (1 - n_ratings_c / total_cells) * 100 if total_cells > 0 else 0
        sparsity = min(sparsity, 99.99)

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card"><div class="number">{n_users:,}</div><div class="label">Total Users</div></div>
            <div class="stat-card"><div class="number">{n_prods:,}</div><div class="label">Total Products</div></div>
            <div class="stat-card"><div class="number">{n_ratings:,}</div><div class="label">Total Ratings</div></div>
            <div class="stat-card"><div class="number orange">{sparsity:.2f}%</div><div class="label">Matrix Sparsity</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card"><div class="number">{n_users_c:,}</div><div class="label">Users (filtered)</div></div>
            <div class="stat-card"><div class="number">{n_prods_c:,}</div><div class="label">Products (filtered)</div></div>
            <div class="stat-card"><div class="number">{n_ratings_c:,}</div><div class="label">Ratings (filtered)</div></div>
            <div class="stat-card"><div class="number">{load_time:.1f}s</div><div class="label">Load Time</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-box">
            <b>Sparsity = {sparsity:.2f}%</b> means that out of {total_cells:,} possible user-product combinations,
            only {n_ratings_c:,} ({100-sparsity:.2f}%) contain actual ratings. This is the central challenge of
            recommendation systems and why we need significance weighting.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        if not HAS_PLOTLY:
            st.error("**Plotly is not installed.** Run `pip install plotly` in your terminal, then restart the app to see interactive charts.")
            st.markdown("---")

            st.markdown("##### Rating Distribution (basic chart)")
            rc = df_clean["Rating"].value_counts().sort_index()
            st.bar_chart(rc)

            st.markdown("##### Ratings per User")
            ua = df_clean.groupby("UserId").size().value_counts().sort_index().head(30)
            st.bar_chart(ua)

            st.markdown("##### Top 10 Most-Rated Products")
            top10 = summary.nlargest(10, "n_ratings").copy()
            top10["name"] = top10["ProductId"].map(id_to_name).fillna(top10["ProductId"])
            st.dataframe(top10[["name", "n_ratings", "avg_rating"]].reset_index(drop=True))

        else:
            # -- Rating Distribution --
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("##### ⭐ Rating Distribution")
                rc = df_clean["Rating"].value_counts().sort_index()
                fig = px.bar(
                    x=rc.index.astype(int), y=rc.values,
                    labels={"x": "Stars", "y": "Count"},
                    color_discrete_sequence=["#FF9900"],
                )
                fig.update_layout(
                    height=340, margin=dict(l=20, r=20, t=10, b=40),
                    plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(tickmode="linear"), bargap=0.3,
                )
                st.plotly_chart(fig, use_container_width=True)
                avg_r = df_clean["Rating"].mean()
                st.caption(f"Average: **{avg_r:.2f}** stars - ratings skew positive (typical e-commerce pattern).")

            with col_b:
                st.markdown("##### 📈 Effective vs Raw Rating")
                sample = df_clean.sample(min(3000, len(df_clean)), random_state=42)
                fig2 = px.scatter(
                    sample, x="Rating", y="EffectiveRating",
                    opacity=0.25,
                    labels={"Rating": "Raw Star Rating", "EffectiveRating": "Effective Rating"},
                    color_discrete_sequence=["#232F3E"],
                )
                fig2.add_shape(type="line", x0=0.5, x1=5.5, y0=0.5, y1=5.5,
                               line=dict(color="#FF9900", dash="dash", width=2))
                fig2.update_layout(
                    height=340, margin=dict(l=20, r=20, t=10, b=40),
                    plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("Points **above** the orange line = positive sentiment boosted the rating. **Below** = negative sentiment.")

            st.divider()

            col_c, col_d = st.columns(2)
            with col_c:
                st.markdown("##### 👥 Ratings per User (Long Tail)")
                user_act = df_clean.groupby("UserId").size().reset_index(name="count")
                fig3 = px.histogram(
                    user_act, x="count", nbins=50,
                    labels={"count": "Ratings per User"},
                    color_discrete_sequence=["#232F3E"],
                )
                fig3.update_layout(
                    height=320, margin=dict(l=20, r=20, t=10, b=40),
                    plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )
                st.plotly_chart(fig3, use_container_width=True)
                med = user_act["count"].median()
                st.caption(f"Median: **{med:.0f}** ratings per user - the long tail creates sparsity.")

            with col_d:
                st.markdown("##### 📦 Ratings per Product (Cold Start)")
                prod_act = df_clean.groupby("ProductId").size().reset_index(name="count")
                fig4 = px.histogram(
                    prod_act, x="count", nbins=50,
                    labels={"count": "Ratings per Product"},
                    color_discrete_sequence=["#FF9900"],
                )
                fig4.update_layout(
                    height=320, margin=dict(l=20, r=20, t=10, b=40),
                    plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )
                st.plotly_chart(fig4, use_container_width=True)
                med_p = prod_act["count"].median()
                st.caption(f"Median: **{med_p:.0f}** ratings per product. Below {min_ratings} = filtered out.")

            st.divider()

            col_e, col_f = st.columns(2)
            with col_e:
                st.markdown("##### 💬 Sentiment Breakdown")
                if text_col:
                    cats = []
                    for ht, sc in zip(df_clean["has_text_review"], df_clean["sentiment_component"]):
                        if not ht:
                            cats.append("No Text Review")
                        elif sc > 0.001:
                            cats.append("Positive")
                        elif sc < -0.001:
                            cats.append("Negative")
                        else:
                            cats.append("Neutral")
                    cat_counts = pd.Series(cats).value_counts()
                    cmap = {"Positive": "#2E7D32", "Negative": "#C62828",
                            "Neutral": "#9E9E9E", "No Text Review": "#BDBDBD"}
                    fig5 = px.pie(
                        names=cat_counts.index, values=cat_counts.values,
                        color=cat_counts.index, color_discrete_map=cmap,
                        hole=0.4,
                    )
                    fig5.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig5, use_container_width=True)
                    st.caption("Sentiment via **Hu & Liu lexicon** (~6,800 opinion words).")
                else:
                    st.info("No text review column detected.")

            with col_f:
                st.markdown("##### 🏆 Top 10 Most-Rated Products")
                top10 = summary.nlargest(10, "n_ratings").copy()
                top10["name"] = top10["ProductId"].map(id_to_name).fillna(top10["ProductId"])
                top10["name"] = top10["name"].apply(lambda x: x[:35]+"..." if len(str(x)) > 35 else x)
                fig6 = px.bar(
                    top10, x="n_ratings", y="name", orientation="h",
                    labels={"n_ratings": "Ratings", "name": "Product"},
                    color_discrete_sequence=["#FF9900"],
                )
                fig6.update_layout(
                    height=320, margin=dict(l=10, r=20, t=10, b=40),
                    plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig6, use_container_width=True)
                st.caption("High-volume products form the backbone of the similarity matrix.")

        st.divider()
        st.markdown("##### Why Sparsity Matters")
        st.markdown(f"""
        The User-Item matrix has **{n_users_c:,} users x {n_prods_c:,} products = {total_cells:,} cells**, but only
        **{n_ratings_c:,}** contain ratings (**{100-sparsity:.2f}%** fill rate).

        **Impact:** Most product pairs share very few common raters, so raw cosine similarity is unreliable.
        Two products sharing a single 5-star rater get 100% similarity — a mathematical artifact, not a real signal.

        **Our mitigations:**
        - **Cold Start Filter**: remove products with < {min_ratings} ratings
        - **Significance Weighting**: similarity x min(common_raters, {SIGNIFICANCE_THRESHOLD}) / {SIGNIFICANCE_THRESHOLD}
        - **Popularity Fallback**: new users get best sellers instead of noisy predictions
        """)

    # -- TAB 4: EVALUATION --
    with tab4:
        st.markdown('<div class="section-header">Model Evaluation</div>', unsafe_allow_html=True)
        st.markdown("""
        We split the data **80% train / 20% test** and measure how well Item-Based CF
        predicts held-out ratings. This validates whether the engine actually works.
        """)

        if st.button("Run Evaluation", key="btn_eval"):
            with st.spinner("Training on 80% and testing on 20%..."):
                results = evaluate_model(df_clean, min_ratings)

            if "error" in results:
                st.warning(results["error"])
            else:
                st.markdown(f"""
                <div class="stat-grid">
                    <div class="stat-card"><div class="number">{results['rmse']:.3f}</div><div class="label">RMSE</div></div>
                    <div class="stat-card"><div class="number">{results['mae']:.3f}</div><div class="label">MAE</div></div>
                    <div class="stat-card"><div class="number orange">{results['precision']:.0%}</div><div class="label">Precision@5</div></div>
                    <div class="stat-card"><div class="number">{results['n_preds']:,}</div><div class="label">Predictions</div></div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="insight-box">
                    Trained on <b>{results['n_train']:,}</b> ratings, tested on <b>{results['n_test']:,}</b> ratings,
                    generated <b>{results['n_preds']:,}</b> predictions.
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")
                st.markdown("##### Interpreting the Results")
                st.markdown(f"""
                | Metric | Value | What it means |
                |--------|-------|---------------|
                | **RMSE** | {results['rmse']:.3f} | Avg prediction error on a 1-5 scale. Netflix Prize winner: ~0.857 |
                | **MAE** | {results['mae']:.3f} | Mean absolute error, more robust to outliers |
                | **Precision@5** | {results['precision']:.0%} | When we predict >= 4 stars, we are right this often |

                **Business implication:** RMSE of {results['rmse']:.2f} means predictions are off by ~{results['rmse']:.1f} stars
                on average. Amazon would A/B test this engine vs the current one and measure revenue per session.
                """)

    # -- TAB 5: ADMIN --
    with tab5:
        st.markdown('<div class="section-header">Admin Panel</div>', unsafe_allow_html=True)

        admin_lookup, admin_add_user, admin_add_review, admin_delete = st.tabs([
            "🔍 Look Up User",
            "➕ Add New User",
            "⭐ Add Review",
            "🗑️ Delete Review",
        ])

        # ── Look Up User ──────────────────────────────────────────────────────
        with admin_lookup:
            st.markdown("Search for any user to see all their reviews.")
            lookup_uid = st.text_input("User ID", key="lookup_uid")
            lookup_btn = st.button("Look Up", key="lookup_btn")

            if lookup_btn:
                uid = lookup_uid.strip()
                if not uid:
                    st.warning("Please enter a User ID.")
                else:
                    master = get_master_df()
                    user_rows = master[master["UserId"].astype(str) == uid]
                    if user_rows.empty:
                        st.error(f"No user found with ID **{uid}**.")
                    else:
                        st.markdown(f'<div class="insight-box">Found <b>{len(user_rows)}</b> review(s) for user <code>{uid}</code>.</div>', unsafe_allow_html=True)
                        for _, row in user_rows.iterrows():
                            pid        = str(row["ProductId"])
                            pname      = id_to_name.get(pid, "")
                            title_line = pname if pname else pid
                            subtitle   = f'<div class="review-row-id">{pid}</div>' if pname else ""
                            rating_val = int(row["Rating"]) if not pd.isna(row["Rating"]) else 0
                            stars      = "⭐" * rating_val + "☆" * (5 - rating_val)
                            rev_text   = str(row.get("Reviews", "") or "")
                            if rev_text in ("nan", ""):
                                rev_text = ""
                            ts = str(row.get("Timestamp", "") or "")
                            if ts == "nan": ts = ""
                            st.markdown(f"""
                            <div class="review-row">
                                <div class="review-row-product">{title_line}</div>
                                {subtitle}
                                <div class="review-row-meta">{stars}&nbsp;•&nbsp;{ts}</div>
                                <div>{rev_text if rev_text else "<em style='color:#aaa'>No review text</em>"}</div>
                            </div>
                            """, unsafe_allow_html=True)

        # ── Add New User ──────────────────────────────────────────────────────
        with admin_add_user:
            st.markdown(
                "Add a new user. The ID must be **exactly 28 characters**, "
                "start with **A**, and use only **uppercase letters and digits** — "
                "matching Amazon's format (e.g. `AEZOOGGU3F75UKWMJOO6HMWSBT3Q`)."
            )

            col_field, col_gen = st.columns([3, 1])
            with col_field:
                new_uid_input = st.text_input(
                    "New User ID", key="new_uid_input",
                    placeholder="e.g. AEZOOGGU3F75UKWMJOO6HMWSBT3Q",
                )
            with col_gen:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Generate ID", key="gen_id_btn"):
                    st.session_state["generated_id"] = generate_amazon_id()

            if "generated_id" in st.session_state:
                st.markdown(f'<div class="insight-box">Generated ID: <code>{st.session_state["generated_id"]}</code> — copy it into the field above.</div>', unsafe_allow_html=True)

            st.markdown("**Assign a first product rating** (required to register the user)")
            new_prod  = st.selectbox("Product", options=product_options, key="new_user_product")
            new_pid   = label_to_id[new_prod]
            new_rate  = st.slider("Rating", 1, 5, 5, key="new_user_rating")
            new_rev   = st.text_area("Review text (optional)", key="new_user_review", height=80)

            if st.button("Add User", key="add_user_btn"):
                uid_val = new_uid_input.strip().upper()
                master  = get_master_df()
                if not uid_val:
                    st.error("Please enter a User ID.")
                elif not is_valid_amazon_id(uid_val):
                    st.error("❌ Not a valid User ID. It must be exactly 28 characters, start with 'A', and contain only uppercase letters and digits.")
                elif uid_val in master["UserId"].astype(str).values:
                    st.warning(f"User **{uid_val}** already exists.")
                else:
                    append_row_to_master({
                        "UserId": uid_val, "ProductId": new_pid,
                        "product_name": id_to_name.get(new_pid, ""),
                        "Rating": float(new_rate), "Reviews": new_rev,
                        "Timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    })
                    st.success(f"✅ User **{uid_val}** added! Look them up in **Look Up User** to verify.")

        # ── Add Review ────────────────────────────────────────────────────────
        with admin_add_review:
            st.markdown("Log a new rating and review for an **existing** user.")

            rev_uid  = st.text_input("User ID", key="review_uid", placeholder="Must be an existing user")
            rev_prod = st.selectbox("Product", options=product_options, key="review_product")
            rev_pid  = label_to_id[rev_prod]
            rev_rate = st.slider("Rating", 1, 5, 5, key="review_rating")
            rev_text = st.text_area("Review text (optional)", key="review_text_inp", height=80)

            if st.button("Submit Review", key="add_review_btn"):
                uid_rev = rev_uid.strip()
                master  = get_master_df()
                if not uid_rev:
                    st.error("Please enter a User ID.")
                elif uid_rev not in master["UserId"].astype(str).values:
                    st.error(f"❌ User **{uid_rev}** does not exist. Add them first in **Add New User**.")
                else:
                    append_row_to_master({
                        "UserId": uid_rev, "ProductId": rev_pid,
                        "product_name": id_to_name.get(rev_pid, ""),
                        "Rating": float(rev_rate), "Reviews": rev_text,
                        "Timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    })
                    st.success(f"✅ Review added for **{uid_rev}**. Check it in **Look Up User**.")
# ── Delete Review ─────────────────────────────────────────────────
        with admin_delete:
            st.markdown("Remove a specific review from an existing user.")

            del_uid_input = st.text_input("User ID", key="del_uid_input", placeholder="Enter user ID to load their reviews")
            load_btn = st.button("Load Reviews", key="load_del_btn")

            if load_btn:
                uid = del_uid_input.strip()
                if not uid:
                    st.warning("Please enter a User ID.")
                else:
                    master = get_master_df()
                    user_rows = master[master["UserId"].astype(str) == uid]
                    if user_rows.empty:
                        st.error(f"No user found with ID **{uid}**.")
                    else:
                        st.session_state["del_user_rows"] = user_rows
                        st.session_state["del_uid_val"] = uid

            if "del_user_rows" in st.session_state:
                user_rows = st.session_state["del_user_rows"]
                uid = st.session_state["del_uid_val"]

                st.markdown(f'<div class="insight-box">Found <b>{len(user_rows)}</b> review(s) for user <code>{uid}</code>. Select one to delete.</div>', unsafe_allow_html=True)

                options = []
                for i, row in user_rows.iterrows():
                    pid    = str(row["ProductId"])
                    pname  = id_to_name.get(pid, pid)
                    rating = int(row["Rating"]) if not pd.isna(row["Rating"]) else 0
                    stars  = "⭐" * rating
                    rev    = str(row.get("Reviews", "") or "")[:50]
                    options.append((i, f"{pname[:40]} — {stars} — {rev}"))

                labels  = [o[1] for o in options]
                indices = [o[0] for o in options]

                selected_label = st.selectbox("Select review to delete", options=labels, key="del_select")
                selected_index = indices[labels.index(selected_label)]

                if st.button("🗑️ Delete this review", key="del_btn"):
                    master = get_master_df()
                    st.session_state["master_df"] = master.drop(index=selected_index).reset_index(drop=True)
                    del st.session_state["del_user_rows"]
                    del st.session_state["del_uid_val"]
                    st.success("✅ Review deleted! Go to **Look Up User** to verify.")
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FOOTER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown(f"""
    <div class="footer">
        <b>TBS Barcelona - Group 7</b> |
        Amazon Product Recommendation System |
        Python - Pandas - Scikit-Learn - Streamlit |
        {n_ratings:,} ratings - {n_prods:,} products - {n_users:,} users
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    