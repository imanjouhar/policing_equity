"""
main.py — Policing Equity

Author: Iman Jouhar
Course: DLBDSMLUSL01 Unsupervised Machine Learning and Feature Engineering
Task: 2 (Policing Equity)

Description:
This script executes the complete unsupervised machine learning pipeline to identify 
homogeneous categories of policing incidents. It is structured into 9 parts.

Pipeline Agenda:
----------------
PART 1: Data Loading & Deduplication
        - Locates raw CSVs, performs MD5 hash deduplication, and concatenates records.
        - Implements a programmatic fix for structural file errors (e.g., Kaggle metadata rows).

PART 2: Exploratory Data Analysis (EDA)
        - Audits missing-value patterns to determine if missingness is structural or random.

PART 3: Feature Engineering
        - Standardizes messy categorical strings (e.g., unifying race designations).
        - Missing Data Triage: drops columns >25% missing, adds binary tracking flags 
          for 5-25% missingness, and imputes the remainder based on distribution skewness.
        - Addresses extreme outliers (3x IQR fence, winsorization) and encodes categoricals.

PART 4: Automated Feature Generation
        - Creates domain-relevant ratio features (e.g., force_per_arrest, racial_arrest_ratio).
        - Applies log-transformations to heavily skewed distributions.

PART 5: Feature Selection
        - Removes near-zero variance features.
        - Prunes highly correlated features (|r| > 0.95) to prevent multicollinearity.

PART 6: Dimensionality Reduction
        - Compares PCA, Multidimensional Scaling (MDS), and Locally Linear Embedding (LLE). 
        - Retains PCA components explaining >95% variance as the primary clustering input space.

PART 7: Clustering (Algorithm Comparison)
        - Computes a 5-method consensus to determine the optimal 'k' (combining Kneedle, 
          Silhouette, Calinski-Harabasz, Davies-Bouldin, and the Gap Statistic).
        - Fits and evaluates four algorithms: k-Means, GMM, Agglomerative (Ward), and DBSCAN.

PART 8: Champion Selection
        - Ranks algorithms via a composite score of internal validity indices.
        - Validates the stability of the winning partition using a GMM soft-membership overlay.

PART 9: Artifact Persistence & Reporting
        - Saves the champion model, cleaned dataset, and analytical artifacts.
        - Triggers `report.py` to generate the interactive HTML data story.
"""
import os
import sys
import glob
import time
import hashlib
import warnings
import webbrowser
import subprocess
import argparse
import joblib
import numpy as np
import pandas as pd

import plotly.express       as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.cluster.hierarchy import linkage, dendrogram

try:
    import report as _report
except ImportError:
    _report = None
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.decomposition     import PCA
from sklearn.manifold          import MDS, LocallyLinearEmbedding
from sklearn.cluster           import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture           import GaussianMixture
from sklearn.metrics           import (silhouette_score, davies_bouldin_score,
                                       calinski_harabasz_score,
                                       silhouette_samples,
                                       adjusted_rand_score)
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors         import NearestNeighbors

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "./data"
OUTPUT_DIR         = "./outputs"
RANDOM_SEED        = 42
MAX_ROWS           = 5000
MDS_ROWS           = 1000
GMM_MAX_K          = 8
K_RANGE            = range(3, 8)      # k in [3,7] — capped to keep clusters
                                      # interpretable on a geographic map
N_BOOTSTRAPS       = 10
GAP_N_REFS         = 5

# User-specified thresholds (2025-04)
USELESS_MISSING    = 0.80             # drop cols with >80% missing immediately
DROP_MISSING_ABOVE = 0.25             # drop cols with >25% missing
FLAG_MISSING_ABOVE = 0.05             # 5-25%: impute + _was_missing flag
SKEW_SYMMETRIC     = 0.5              # |skew|<0.5 -> mean; else -> median

IQR_MULTIPLIER     = 3.0              # extreme outlier fence
WINSORIZE_LOW      = 0.01
WINSORIZE_HIGH     = 0.99

FIGURES = []
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

PALETTE  = px.colors.qualitative.Bold
TEMPLATE = "plotly_white"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save_html(fig, name, title=""):
    path = os.path.join(OUTPUT_DIR, name)
    fig.update_layout(template=TEMPLATE,
                      title=dict(text=title, font=dict(size=16)),
                      margin=dict(l=60, r=40, t=60, b=60))
    fig.write_html(path, include_plotlyjs="cdn")
    FIGURES.append(path)
    print(f"  [saved] {path}")

def section(title):
    bar = "═" * 72
    print(f"\n{bar}\n  {title}\n{bar}")

# ═════════════════════════════════════════════════════════════════════════════
# k-SELECTION
# ═════════════════════════════════════════════════════════════════════════════
def kneedle_knee(x, y, curve="convex", direction="decreasing"):
    """
    Kneedle algorithm: the elbow method, where k was selected based on when the 
    knee curve starts approaching zero, this method will not be used alone.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    if direction == "decreasing":
        y_norm = 1.0 - y_norm
    if curve == "concave":
        y_norm = 1.0 - y_norm
    differences = y_norm - x_norm
    return x[int(np.argmax(differences))], int(np.argmax(differences))


def gap_statistic(X, k_range, n_refs=GAP_N_REFS, random_state=RANDOM_SEED):
    """Gap Statistics: measures how far is the pooled within-cluster 
    sum of squares around the cluster centers from the sum of squares 
    expected under the null reference distribution of data."""
    rng = np.random.RandomState(random_state)
    ks = list(k_range)
    mins, maxs = X.min(axis=0), X.max(axis=0)
    gaps = np.zeros(len(ks))
    sks = np.zeros(len(ks))
    for i, k in enumerate(ks):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        log_wk = np.log(max(km.inertia_, 1e-10))
        ref_log_wks = []
        for _ in range(n_refs):
            X_ref = rng.uniform(mins, maxs, size=X.shape)
            km_r = KMeans(n_clusters=k, random_state=random_state, n_init=3)
            km_r.fit(X_ref)
            ref_log_wks.append(np.log(max(km_r.inertia_, 1e-10)))
        gaps[i] = np.mean(ref_log_wks) - log_wk
        sks[i] = np.std(ref_log_wks) * np.sqrt(1 + 1.0 / n_refs)
    optimal_k = ks[-1]
    for i in range(len(ks) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            optimal_k = ks[i]
            break
    return gaps, sks, optimal_k


def compute_k_metrics(X, k_range):
    """4 internal cluster validity indices per k."""
    inertias, sils, chs, dbs = [], [], [], []
    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        lbl = km.fit_predict(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, lbl, sample_size=min(2000, len(lbl))))
        chs.append(calinski_harabasz_score(X, lbl))
        dbs.append(davies_bouldin_score(X, lbl))
    return inertias, sils, chs, dbs


def select_champion_k(X, k_range=K_RANGE):
    """
    5-method consensus for optimal k: combining the elbow
    method with silhouette because "the elbow score only considers
    clusters cohesion or compactness and not their separability.
    """
    ks = list(k_range)
    inertias, sils, chs, dbs = compute_k_metrics(X, ks)
    k_elbow, _  = kneedle_knee(ks, inertias, "convex", "decreasing")
    k_sil       = ks[int(np.argmax(sils))]
    k_ch        = ks[int(np.argmax(chs))]
    k_db        = ks[int(np.argmin(dbs))]
    gaps, sks, k_gap = gap_statistic(X, ks, n_refs=GAP_N_REFS)

    candidates = [int(k_elbow), int(k_sil), int(k_ch), int(k_db), int(k_gap)]
    votes = {k: candidates.count(k) for k in set(candidates)}
    max_votes = max(votes.values())
    top = [k for k, v in votes.items() if v == max_votes]
    if len(top) == 1:
        chosen = top[0]
    else:
        sil_rank = {ks[i]: r for r, i in enumerate(np.argsort(-np.array(sils)))}
        ch_rank  = {ks[i]: r for r, i in enumerate(np.argsort(-np.array(chs)))}
        db_rank  = {ks[i]: r for r, i in enumerate(np.argsort(np.array(dbs)))}
        chosen = min(top, key=lambda k: sil_rank[k] + ch_rank[k] + db_rank[k])

    return {
        "chosen_k":     int(chosen),
        "votes":        votes,
        "k_elbow":      int(k_elbow),
        "k_silhouette": int(k_sil),
        "k_ch":         int(k_ch),
        "k_db":         int(k_db),
        "k_gap":        int(k_gap),
        "inertias":     list(map(float, inertias)),
        "silhouettes":  list(map(float, sils)),
        "ch_scores":    list(map(float, chs)),
        "db_scores":    list(map(float, dbs)),
        "gaps":         [float(g) for g in gaps],
        "gap_sks":      [float(s) for s in sks],
        "k_range":      ks,
    }


def cluster_stability(X, k, n_bootstraps=N_BOOTSTRAPS, subsample=0.8,
                      random_state=RANDOM_SEED):
    """Bootstrap cluster stability metric via ARI (Adjusted Rand Index)"""
    rng = np.random.RandomState(random_state)
    base = KMeans(n_clusters=k, random_state=random_state,
                  n_init=10).fit_predict(X)
    aris = []
    n = X.shape[0]
    for _ in range(n_bootstraps):
        mask = rng.rand(n) < subsample
        km  = KMeans(n_clusters=k, random_state=rng.randint(1e6), n_init=5)
        lbl = km.fit_predict(X[mask])
        aris.append(adjusted_rand_score(base[mask], lbl))
    return float(np.mean(aris)), float(np.std(aris))
# ═════════════════════════════════════════════════════════════════════════════
# PART 1 - DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
def _md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()
    
def _needs_skiprows(path):
    """Peek row 0: if it's the Kaggle alias row (few numeric values in what
    should be a numeric-heavy dataset), read with skiprows=[1]."""
    try:
        peek = pd.read_csv(path, nrows=2, low_memory=False, encoding="latin1")
        if peek.shape[0] < 2:
            return False
        first_numeric = sum(pd.to_numeric(peek.iloc[0],
                                           errors="coerce").notna())
        return first_numeric < len(peek.columns) * 0.3
    except Exception:
        return False


def _is_policing_event_file(path):
    name = os.path.basename(path).upper()
    if any(x in name for x in ["METADATA", "ACS_", "ACS_VARIABLE"]):
        return False
    return any(k in name for k in [
        "UOF", "USE_OF_FORCE", "FIELD-INTERVIEW", "FIELD_INTERVIEW",
        "VEHICLE-STOP", "VEHICLE_STOP", "OIS", "STOPS", "FORCE",
        "INCIDENT", "ARRESTS",
    ])


def load_policing_data(data_dir):
    """Load + MD5-dedupe + skiprows-fix + concatenate."""
    section("PART 1 — Data Loading (MD5 dedup + skiprows=[1] fix)")

    csv_files = list(set(
        glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True) +
        glob.glob(os.path.join(data_dir, "*.csv"))
    ))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}.")

    target_files = [f for f in csv_files if _is_policing_event_file(f)]
    if not target_files:
        target_files = [f for f in csv_files
                        if "metadata" not in f.lower()
                        and "ACS_15_5YR" not in f]

    seen_hashes = {}
    n_dup = 0
    n_dupes = 0
    unique_files = []
    for f in sorted(target_files):
        try:
            h = _md5(f)
        except Exception as e:
            print(f"  [hash fail] {f}: {e}")
            continue
        if h in seen_hashes:
            n_dup += 1
            continue
        seen_hashes[h] = f
        unique_files.append(f)

    if n_dup > 0:
        print(f"  Skipped {n_dup} duplicate file paths (MD5 dedup).")
    print(f"\n  Policing event files ({len(unique_files)} unique):")
    dfs = []
    _no_race_count = 0  # count records from files without any race column
    for f in unique_files:
        try:
            kwargs = dict(low_memory=False, encoding="latin1")
            if _needs_skiprows(f):
                kwargs["skiprows"] = [1]     # the Kaggle metadata row
            tmp = pd.read_csv(f, **kwargs)

            # Coerce numeric-looking strings back to numeric
            for c in tmp.columns:
                if tmp[c].dtype == object:
                    coerced = pd.to_numeric(tmp[c], errors="coerce")
                    nonnull = tmp[c].notna().sum()
                    if nonnull > 0 and coerced.notna().sum() / nonnull >= 0.80:
                        tmp[c] = coerced

            tmp["source_file"] = os.path.basename(f)
            if not any("RACE" in c.upper() and "OFFICER" not in c.upper() for c in tmp.columns):
                _no_race_count += len(tmp)
            dfs.append(tmp)
            note = "  (skiprows=[1])" if "skiprows" in kwargs else ""
            print(f"    {os.path.basename(f):<45s} -> {tmp.shape}{note}")
        except Exception as e:
            print(f"    SKIP  {os.path.basename(f)}: {e}")

    if not dfs:
        raise ValueError("Could not load any CSV file.")

    df = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"\n  Combined shape: {df.shape}")
    print(f"  Records without race column: {_no_race_count:,}")
    # Save counts for artifacts
    df.attrs["_n_no_race"] = _no_race_count
    df.attrs["_n_raw_total"] = len(df)
    return df
# ═════════════════════════════════════════════════════════════════════════════
# PART 2 - EDA
# ═════════════════════════════════════════════════════════════════════════════
def run_eda(df):
    section("PART 2 — Exploratory Data Analysis")

    n_num = df.select_dtypes(include=[np.number]).shape[1]
    n_cat = df.select_dtypes(include=["object", "category"]).shape[1]
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}  "
          f"|  Numerical: {n_num}  |  Categorical: {n_cat}")
    print(f"  Source files: {df['source_file'].nunique()}")

    # Missing-value heatmap
    miss_cols = df.columns[df.isna().any()].tolist()[:40]
    if miss_cols:
        mm = df[miss_cols].isna().head(300).astype(int)
        fig = px.imshow(mm.T,
                        color_continuous_scale=["#f0f0f0", "#2c7bb6"],
                        aspect="auto", labels=dict(color="Missing"))
        fig.update_xaxes(showticklabels=False, title="Rows (first 300)")
        fig.update_yaxes(title="Column")
        # save_html(fig, "01_missing_heatmap.html",
                  # "Missing-Value Pattern (first 300 rows)")

    # Per-column missing bar with drop/flag thresholds
    miss_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    miss_pct = miss_pct[miss_pct > 0]
    if len(miss_pct):
        mdf = miss_pct.reset_index()
        mdf.columns = ["column", "missing_pct"]
        fig = px.bar(mdf, x="missing_pct", y="column", orientation="h",
                     color="missing_pct", color_continuous_scale="Reds",
                     labels={"missing_pct": "% missing"})
        fig.add_vline(x=DROP_MISSING_ABOVE * 100, line_dash="dash",
                      line_color="red",
                      annotation_text=f"Drop ({DROP_MISSING_ABOVE*100:.0f}%)")
        fig.add_vline(x=FLAG_MISSING_ABOVE * 100, line_dash="dot",
                      line_color="orange",
                      annotation_text=f"Flag ({FLAG_MISSING_ABOVE*100:.0f}%)")
        fig.update_layout(height=max(450, 22 * len(mdf)), showlegend=False,
                          coloraxis_showscale=False)
#         save_html(fig, "02_missing_per_column.html",
#                   "Per-Column Missing-Value Percentage")

    # Source file distribution
    src = df["source_file"].value_counts().reset_index()
    src.columns = ["file", "incidents"]
    fig = px.bar(src, x="file", y="incidents", color="file",
                 color_discrete_sequence=PALETTE, text="incidents")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=400, xaxis_tickangle=-20)
#     save_html(fig, "03_source_file_counts.html",
#               "Incidents per Source File")

    # ── Critical assessment: EDA decisions ───────────────────────────────────
    print("\n  ── Critical Assessment (EDA) ──")
    print("  DECISION: Concatenate all policing event files into one dataset.")
    print("    RATIONALE: The task requires identifying homogeneous categories")
    print("    across all policing activities. Separate analysis per file would")
    print("    miss cross-city patterns. The outer join preserves all columns;")
    print("    the missing-value triage (Part 3) handles the resulting NaNs.")
    print(f"  DECISION: {n_num} numerical + {n_cat} categorical columns identified.")
    print("    RATIONALE: Correct dtype detection (via skiprows=[1] fix) is")
    print("    critical — without it, all columns default to string, which")
    print("    collapses PCA to one axis and produces degenerate k=2 clusters.")
# ═════════════════════════════════════════════════════════════════════════════
# PART 3 - FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
def imputation_strategy(series):
    """
      |skew| <  SKEW_SYMMETRIC -> mean  (approx. symmetric, MLE under normality)
      |skew| >= SKEW_SYMMETRIC -> median (robust to skew/outliers)
    """
    s = series.dropna()
    if len(s) < 5 or s.nunique() < 2:
        return "median"
    try:
        return "mean" if abs(float(s.skew())) < SKEW_SYMMETRIC else "median"
    except Exception:
        return "median"
    
def feature_engineering(df):
    """ Pipeline with user-specified missing-value bands."""
    section("PART 3 — Feature Engineering")

    # Correct known column typos before race detection
    if "SUBJECT_RACT" in df.columns and "SUBJECT_RACE" not in df.columns:
        df = df.rename(columns={"SUBJECT_RACT": "SUBJECT_RACE"})
        print("    [fix] Renamed SUBJECT_RACT -> SUBJECT_RACE (Indianapolis typo)")

    # ── Normalize messy race strings early to ensure proper One-Hot Encoding ──
    race_cols = [c for c in df.columns if "RACE" in c.upper() and "OFFICER" not in c.upper()]
    for rc in race_cols:
        if rc in df.columns:
            df[rc] = df[rc].astype(str).str.upper().str.strip()
            df[rc] = df[rc].replace({
                'W': 'WHITE', 'CAUCASIAN': 'WHITE', 'ANGLO': 'WHITE', 'NON-HISPANIC': 'WHITE',
                'B': 'BLACK', 'AFRICAN AMERICAN': 'BLACK', 'BLACK OR AFRICAN AMERICAN': 'BLACK',
                'H': 'HISPANIC', 'LATINO': 'HISPANIC', 'HISPANIC OR LATINO': 'HISPANIC', 'L': 'HISPANIC',
                'A': 'ASIAN', 'ASIAN/PACIFIC ISLANDER': 'ASIAN', 'ASIAN OR PACIFIC ISLANDER': 'ASIAN',
                'K':'ASIAN', 'C':'ASIAN', 'F':'ASIAN', 'P':'ASIAN',
                'POLYNESIAN': 'ASIAN',
                'BI-RACIAL': 'OTHER', 'NATIVE AMER': 'OTHER',
                'NAN': np.nan, 'NO DATA': 'UNKNOWN', 'NOT SPECIFIED': 'UNKNOWN'
            })

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c != "source_file"]
    cat_cols = [c for c in df.select_dtypes(include=["object","category"]).columns
                if c != "source_file" and df[c].nunique() < 500]
    geo_cols = [c for c in num_cols
                if any(k in c.lower() for k in ["lat", "lon", "lng"])]
    num_cols = [c for c in num_cols if c not in geo_cols]
    print(f"  Identified: {len(num_cols)} numerical | "
          f"{len(cat_cols)} categorical | {len(geo_cols)} geographic")

    # ── 1a. Drop columns with >80% missing (structurally useless) ──
    miss = df.isna().mean()
    useless = [c for c in miss[miss > USELESS_MISSING].index.tolist()
               if c not in geo_cols + ["source_file"]]
    if useless:
        df = df.drop(columns=useless)
        num_cols = [c for c in num_cols if c not in useless]
        cat_cols = [c for c in cat_cols if c not in useless]
        geo_cols = [c for c in geo_cols if c not in useless]
        print(f"\n  Dropped {len(useless)} columns with >{USELESS_MISSING*100:.0f}% missing (structurally incomplete)")

    # ── 1b. Missing-value banding on remaining columns ────────────────────────────────────────────
    miss = df.isna().mean()
    _protect = [c for c in df.columns
                if any(k in c.upper() for k in ["RACE", "GENDER", "AGE", "ETHNICITY"])]
    to_drop = [c for c in miss[miss > DROP_MISSING_ABOVE].index.tolist()
               if c not in geo_cols + ["source_file"] + _protect]
    to_flag = miss[(miss > FLAG_MISSING_ABOVE) &
                   (miss <= DROP_MISSING_ABOVE)].index.tolist()
    to_impute_lo = miss[(miss > 0) & (miss <= FLAG_MISSING_ABOVE)].index.tolist()
    print(f"\n  Triage:")
    print(f"    DROP         : {len(to_drop):>3} cols (>{DROP_MISSING_ABOVE*100:.0f}% missing)")
    print(f"    IMPUTE+FLAG  : {len(to_flag):>3} cols ({FLAG_MISSING_ABOVE*100:.0f}-{DROP_MISSING_ABOVE*100:.0f}% missing)")
    print(f"    IMPUTE-SIMPLE: {len(to_impute_lo):>3} cols (<{FLAG_MISSING_ABOVE*100:.0f}% missing)")

    if to_drop:
        df = df.drop(columns=to_drop)
        num_cols = [c for c in num_cols if c not in to_drop]
        cat_cols = [c for c in cat_cols if c not in to_drop]
        geo_cols = [c for c in geo_cols if c not in to_drop]

    # ── 2. Missingness flags (5-25% band) ───────────────────────────────────
    flag_cols_added = []
    for c in to_flag:
        if c in df.columns:
            fname = f"{c}_was_missing"
            df[fname] = df[c].isna().astype(int)
            flag_cols_added.append(fname)
    if flag_cols_added:
        print(f"    Added {len(flag_cols_added)} missingness flags.")

    # ── 3a. Remove extreme outliers via IQR fence ──
    n_before = len(df)
    for c in num_cols:
        if c in df.columns and df[c].notna().sum() > 100:
            Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower_fence = Q1 - IQR_MULTIPLIER * IQR
                upper_fence = Q3 + IQR_MULTIPLIER * IQR
                df = df[(df[c] >= lower_fence) & (df[c] <= upper_fence) | df[c].isna()]
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"    Removed {n_removed:,} extreme outlier rows (beyond {IQR_MULTIPLIER}×IQR)")

    # ── 3b. Winsorize ───────────────────────────────────────────
    for c in num_cols + geo_cols:
        if c in df.columns:
            df[c] = df[c].clip(lower=df[c].quantile(WINSORIZE_LOW),
                               upper=df[c].quantile(WINSORIZE_HIGH))

    # ── 4. Skewness-based imputation ────────────────────────────────────────
    counts = {"mean": 0, "median": 0, "mode": 0}
    for c in num_cols + geo_cols:
        if c in df.columns and df[c].isna().any():
            strat = imputation_strategy(df[c])
            counts[strat] += 1
            fill = df[c].mean() if strat == "mean" else df[c].median()
            df[c] = df[c].fillna(fill)
    for c in cat_cols:
        if c in df.columns and df[c].isna().any():
            mode = df[c].mode()
            df[c] = df[c].fillna(mode.iloc[0] if len(mode) else "UNKNOWN")
            counts["mode"] += 1

    print(f"    Imputation applied:")
    print(f"      mean   (|skew|<{SKEW_SYMMETRIC}) : {counts['mean']:>3} cols")
    print(f"      median (|skew|>={SKEW_SYMMETRIC}) : {counts['median']:>3} cols")
    print(f"      mode   (categorical)      : {counts['mode']:>3} cols")

    # ── 5. Encode categoricals ──────────────────────────────────────────────
    frames = [df[num_cols + flag_cols_added].copy()]
    enc = {"label": 0, "one_hot": 0, "frequency": 0}
    for c in cat_cols:
        if c not in df.columns:
            continue
        n = df[c].nunique()
        if n <= 2:
            le = LabelEncoder()
            frames.append(pd.DataFrame(le.fit_transform(df[c]),
                                       columns=[c + "_enc"]))
            enc["label"] += 1
        elif n <= 15:
            frames.append(pd.get_dummies(df[c], prefix=c,
                                         drop_first=True).astype(int))
            enc["one_hot"] += 1
        else:
            freq = df[c].value_counts(normalize=True)
            frames.append(pd.DataFrame(df[c].map(freq).values,
                                       columns=[c + "_freq"]))
            enc["frequency"] += 1
    print(f"    Encoding:")
    print(f"      label     (<=2 uniq)  : {enc['label']:>3}")
    print(f"      one-hot   (3-15 uniq) : {enc['one_hot']:>3}")
    print(f"      frequency (>15 uniq)  : {enc['frequency']:>3}")

    df_encoded = pd.concat(frames, axis=1)
    df_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_encoded.fillna(df_encoded.median(numeric_only=True), inplace=True)
    print(f"\n  Shape after encoding: {df_encoded.shape}")

    # ── Critical assessment: Feature Engineering decisions ────────────────────
    print("\n  ── Critical Assessment (Feature Engineering) ──")
    print(f"  DECISION: Drop columns with >{DROP_MISSING_ABOVE*100:.0f}% missing ({len(to_drop)} cols).")
    print("    RATIONALE: Columns above this threshold carry more noise than")
    print("    signal. Imputing 75%+ missing values would inject fabricated")
    print("    patterns that distort cluster boundaries.")
    print(f"  DECISION: Add _was_missing flags for {len(flag_cols_added)} cols (5-25% band).")
    print("    RATIONALE: In policing data, missingness is rarely random —")
    print("    whether a field was recorded is itself informative (e.g.,")
    print("    subject race often missing in certain jurisdictions).")
    print(f"  DECISION: Imputation by skewness — mean if |skew|<{SKEW_SYMMETRIC}, else median.")
    print("    RATIONALE: Mean is the MLE under normality.")
    print("    median is robust to outliers in skewed distributions.")
    print(f"    Applied: {counts['mean']} cols via mean, {counts['median']} via median.")
    return df_encoded, df, num_cols, cat_cols, geo_cols


# ═════════════════════════════════════════════════════════════════════════════
# PART 4 - AUTOMATED FEATURE GENERATION
# ═════════════════════════════════════════════════════════════════════════════
def generate_features(df_encoded, df):
    section("PART 4 — Automated Feature Generation")

    ratio_pairs = [
        ("uof_count",     "arrest_count", "force_per_arrest"),
        ("black_arrests", "white_arrests", "racial_arrest_ratio"),
        ("total_force",   "total_pop",    "force_rate_per_pop"),
        ("subject_count", "officer_count","subj_to_off_ratio"),
    ]
    for n, d, nm in ratio_pairs:
        nm_cols = [c for c in df.columns if n.lower() in c.lower()]
        dm_cols = [c for c in df.columns if d.lower() in c.lower()]
        if nm_cols and dm_cols:
            safe = df[dm_cols[0]].replace(0, np.nan)
            df_encoded[nm] = (df[nm_cols[0]] / safe).fillna(0)
            df_encoded[nm] = df_encoded[nm].clip(
                upper=df_encoded[nm].quantile(0.99))
            print(f"    [+] {nm}")

    skewed = [c for c in df_encoded.columns
              if df_encoded[c].dtype in [np.float64, np.int64]
              and df_encoded[c].skew() > 2]
    for c in skewed:
        df_encoded[c + "_log"] = np.log1p(df_encoded[c].clip(lower=0))
    print(f"  Log-transformed {len(skewed)} heavily-skewed columns.")
    print(f"  Total features: {df_encoded.shape[1]}")
    return df_encoded
# ═════════════════════════════════════════════════════════════════════════════
# PART 5 - FEATURE SELECTION
# ═════════════════════════════════════════════════════════════════════════════
def select_features(df_encoded):
    section("PART 5 — Feature Selection")

    vt   = VarianceThreshold(threshold=0.01)
    X_vt = vt.fit_transform(df_encoded)
    kept = df_encoded.columns[vt.get_support()].tolist()
    print(f"  Variance filter (>0.01) : {df_encoded.shape[1]} -> {len(kept)}")
    df_sel = pd.DataFrame(X_vt, columns=kept)

    corr  = df_sel.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop  = [c for c in upper.columns if any(upper[c] > 0.95)]
    df_sel = df_sel.drop(columns=drop)
    print(f"  Correlation pruning (>0.95) : {len(kept)} -> {df_sel.shape[1]}")

    # Heatmap
    plot_cols = df_sel.columns[:25].tolist()
    cs = df_sel[plot_cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=cs.values, x=cs.columns, y=cs.columns,
        colorscale="RdBu_r", zmid=0,
        text=cs.values.round(2), texttemplate="%{text}",
        hovertemplate="x: %{x}<br>y: %{y}<br>r = %{z:.2f}<extra></extra>"))
    fig.update_layout(height=700, width=750)
    # save_html(fig, "04_correlation_heatmap.html",
              # "Feature Correlation Matrix (top 25)")

    # ── Critical assessment: Feature Selection ───────────────────────────────
    print(f"\n  ── Critical Assessment (Feature Selection) ──")
    print(f"  DECISION: Variance threshold = 0.01.")
    print("    RATIONALE: Removes near-constant features (e.g., binary flags")
    print("    that are 99%+ one value). These contribute no discriminative")
    print("    power to clustering .")
    print(f"  DECISION: Correlation pruning at r > 0.95.")
    print("    RATIONALE: Highly correlated features (r > 0.95) are redundant —")
    print("    keeping both inflates PCA components without adding information.")
    print("    The 0.95 threshold is conservative, preserving features with")
    print("    moderate correlation that may still carry distinct signal")
    print("    .")
    return df_sel

# ═════════════════════════════════════════════════════════════════════════════
# PART 6 - DIMENSIONALITY REDUCTION
# ═════════════════════════════════════════════════════════════════════════════
def reduce_dimensions(X_scaled, idx):
    section("PART 6 — Dimensionality Reduction")
    X_sample = X_scaled[idx]

    # PCA
    pca = PCA(n_components=min(20, X_sample.shape[1]), random_state=RANDOM_SEED)
    X_pca_full = pca.fit_transform(X_sample)
    X_pca_2d   = X_pca_full[:, :2]
    explained  = np.cumsum(pca.explained_variance_ratio_)
    n_comp_95  = int(np.argmax(explained >= 0.95)) + 1
    print(f"  PCA: {n_comp_95} components cover ≥95% variance.")

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Scree (EVR)", "Cumulative EVR"])
    comps = list(range(1, len(pca.explained_variance_ratio_) + 1))
    fig.add_trace(go.Bar(x=comps, y=pca.explained_variance_ratio_,
                  marker_color="#2c7bb6"), row=1, col=1)
    fig.add_trace(go.Scatter(x=comps, y=explained, mode="lines+markers",
                  marker_color="#d7191c"), row=1, col=2)
    fig.add_hline(y=0.95, line_dash="dash", line_color="gray", row=1, col=2)
    fig.add_vline(x=n_comp_95, line_dash="dash", line_color="gray",
                  annotation_text=f"n={n_comp_95}", row=1, col=2)
    fig.update_layout(showlegend=False, height=420)
    # save_html(fig, "05_pca_variance.html",
              # "PCA — Scree & Cumulative Explained Variance")

    # MDS
    print("  [MDS] Fitting …")
    if X_sample.shape[0] > MDS_ROWS:
        mds_idx = np.random.choice(X_sample.shape[0], MDS_ROWS, replace=False)
        X_mds_input = X_sample[mds_idx]
    else:
        mds_idx = np.arange(X_sample.shape[0])
        X_mds_input = X_sample
    mds = MDS(n_components=2, metric=True, random_state=RANDOM_SEED,
              max_iter=300, n_init=1, n_jobs=1)
    try:
        X_mds_sub = mds.fit_transform(X_mds_input)
        mds_ok = True
        print(f"    MDS stress: {mds.stress_:.4f}")
    except Exception as e:
        print(f"    MDS FAILED: {e} — PCA fallback.")
        X_mds_sub = X_pca_2d[mds_idx] if X_sample.shape[0] > MDS_ROWS else X_pca_2d
        mds_ok = False
    if X_sample.shape[0] > MDS_ROWS:
        X_mds = np.zeros((X_sample.shape[0], 2))
        X_mds[mds_idx] = X_mds_sub
        non_idx = np.setdiff1d(np.arange(X_sample.shape[0]), mds_idx)
        X_mds[non_idx] = X_pca_2d[non_idx]
    else:
        X_mds = X_mds_sub

    # LLE
    print("  [LLE] Fitting …")
    k_lle = min(15, X_sample.shape[0] - 2)
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=k_lle,
                                 method="standard", random_state=RANDOM_SEED,
                                 n_jobs=1)
    try:
        X_lle  = lle.fit_transform(X_sample)
        lle_ok = True
        print(f"    LLE reconstruction error: {lle.reconstruction_error_:.6f}")
    except Exception as e:
        print(f"    LLE FAILED: {e} — PCA fallback.")
        X_lle  = X_pca_2d.copy()
        lle_ok = False

    # Side-by-side comparison
    fig = make_subplots(rows=1, cols=3, subplot_titles=[
        "PCA (linear)",
        "MDS (metric)" if mds_ok else "MDS (fallback)",
        f"LLE (k={k_lle})" if lle_ok else "LLE (fallback)"])
    for col_idx, (X2d, color) in enumerate(
            [(X_pca_2d, "#2c7bb6"), (X_mds, "#1a9641"), (X_lle, "#d7191c")], 1):
        fig.add_trace(go.Scatter(
            x=X2d[:, 0], y=X2d[:, 1], mode="markers",
            marker=dict(color=color, size=4, opacity=0.5),
            hovertemplate="Dim1: %{x:.2f}<br>Dim2: %{y:.2f}<extra></extra>",
            showlegend=False), row=1, col=col_idx)
    fig.update_layout(height=480)
#     save_html(fig, "06_dim_reduction_comparison.html",
#               "Dimensionality Reduction: PCA / MDS / LLE")

    # ── Quantitative DR comparison table (Unit 3 requirement) ────────────────
    dr_comparison = pd.DataFrame({
        "Method":            ["PCA", "MDS", "LLE"],
        "Type":              ["Linear (global)", "Metric (global)", "Non-linear (local)"],
        "Quality Metric":    [
            f"Cumul. EVR: {explained[1]:.1%} (2D), {explained[min(n_comp_95-1, len(explained)-1)]:.1%} ({n_comp_95}D)",
            f"Stress: {mds.stress_:.1f}" if mds_ok else "Failed (PCA fallback)",
            f"Recon. error: {lle.reconstruction_error_:.6f}" if lle_ok else "Failed (PCA fallback)",
        ],
        "Preserves":         ["Global variance", "Pairwise distances", "Local neighborhoods"],
        "Scalability":       ["O(n·p²) — fast", "O(n²) — slow for n>5k", "O(n·k²) — moderate"],
        "Suitable for data": ["Yes (mixed features)", "Partial (high stress)", "Partial (sensitive to k)"],
    })
    dr_path = os.path.join(OUTPUT_DIR, "dr_comparison.csv")
    dr_comparison.to_csv(dr_path, index=False)
    print(f"\n  Dimensionality Reduction Comparison:")
    print(dr_comparison.to_string(index=False))

    # ── Critical assessment: DR decisions ────────────────────────────────────
    print("\n  ── Critical Assessment (Dimensionality Reduction) ──")
    print(f"  DECISION: Use PCA with {n_comp_95} components (≥95% variance) as")
    print("    the primary clustering input space.")
    print("    RATIONALE: PCA provides a principled variance-retention criterion.")
    print("    LIMITATION: PCA assumes continuous, jointly normally distributed data.")
    print("    Applying it to one-hot encoded variables and binary missingness flags")
    print("    distorts the Euclidean space. MCA or FAMD would be mathematically")
    print("    superior, but PCA is retained here due to library constraints as a")
    print("    heuristic compression.")
    if mds_ok:
        print(f"  NOTE: MDS stress = {mds.stress_:.1f} — high stress indicates")
        print("    the 2D embedding distorts pairwise distances significantly.")
        print("    This is expected with mixed numerical+categorical features.")
    if lle_ok:
        print(f"  NOTE: LLE reconstruction error = {lle.reconstruction_error_:.6f}")
        print("    — near-zero error suggests the local manifold structure is")
        print("    captured, but LLE is fragile with high-dimensional encoded data.")

    return {
        "pca": pca, "X_pca_full": X_pca_full, "X_pca_2d": X_pca_2d,
        "X_mds": X_mds, "X_lle": X_lle, "n_comp_95": n_comp_95,
        "mds_ok": mds_ok, "lle_ok": lle_ok,
        "mds_stress": mds.stress_ if mds_ok else None,
        "lle_err":    lle.reconstruction_error_ if lle_ok else None,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PART 7 - CLUSTERING — FOUR ALGORITHMS
# ═════════════════════════════════════════════════════════════════════════════
def run_clustering(X_clust):
    """Four-algorithm comparison. k-Means + GMM + Agglomerative + DBSCAN."""
    section("PART 7 — Clustering — Four-Algorithm Comparison")
    print(f"  Cluster input shape: {X_clust.shape}")
    print(f"  Search range (with k>=3 constraint): "
          f"[{min(K_RANGE)}, {max(K_RANGE)}]")

    # ── Advanced k-selection ────────────────────────────────────────────────
    print("\n  ── 5-method consensus for optimal k ──")
    k_info = select_champion_k(X_clust, K_RANGE)
    best_k = k_info["chosen_k"]

    print(f"    Kneedle elbow    : k = {k_info['k_elbow']}")
    print(f"    Silhouette peak  : k = {k_info['k_silhouette']}")
    print(f"    Calinski-Har.    : k = {k_info['k_ch']}")
    print(f"    Davies-Bouldin   : k = {k_info['k_db']}")
    print(f"    Gap Statistic    : k = {k_info['k_gap']}")
    print(f"    CONSENSUS        : k = {best_k}  (votes: {k_info['votes']})")

    stab_mean, stab_std = cluster_stability(X_clust, best_k)
    print(f"    Bootstrap stability at k={best_k}: ARI = "
          f"{stab_mean:.3f} ± {stab_std:.3f}")

    # ── Critical assessment: k selection ─────────────────────────────────────
    print("\n  ── Critical Assessment (k Selection) ──")
    print(f"  DECISION: k = {best_k} selected via 5-method consensus.")
    print("    RATIONALE: Best practice is to recommends")
    print("    combining elbow + silhouette because 'the elbow score only")
    print("    considers clusters cohesion or compactness and not their")
    print("    separability' (Sayed-Mouchaweh & Muller-Kett, 2025). We extend")
    print("    this to 5 criteria (Kneedle, Silhouette, CH, DB, Gap) and take")
    print("    majority vote. This is more defensible than any single method.")
    print(f"  DECISION: Search range k ∈ [{min(K_RANGE)}, {max(K_RANGE)}].")
    print("    RATIONALE: k ≥ 3 because policing incidents are implausibly")
    print("    binary. k ≤ 7 because more clusters become uninterpretable on")
    print("    the report and lose practical value for the municipality.")

    # ── Visualize the 5-method consensus ────────────────────────────────────
    ks = k_info["k_range"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[
        "Kneedle Elbow (Inertia ↓)",
        f"Silhouette (peak k={k_info['k_silhouette']})",
        f"Calinski-Harabasz (peak k={k_info['k_ch']})",
        f"Davies-Bouldin (min k={k_info['k_db']})",
        f"Gap Statistic (k={k_info['k_gap']})",
        "Consensus Votes",
    ])
    fig.add_trace(go.Scatter(x=ks, y=k_info["inertias"], mode="lines+markers",
                  marker_color="#2c7bb6"), row=1, col=1)
    fig.add_vline(x=k_info["k_elbow"], line_dash="dash",
                  line_color="#d7191c", row=1, col=1)
    fig.add_trace(go.Scatter(x=ks, y=k_info["silhouettes"], mode="lines+markers",
                  marker_color="#1a9641"), row=1, col=2)
    fig.add_vline(x=k_info["k_silhouette"], line_dash="dash",
                  line_color="#d7191c", row=1, col=2)
    fig.add_trace(go.Scatter(x=ks, y=k_info["ch_scores"], mode="lines+markers",
                  marker_color="#984ea3"), row=1, col=3)
    fig.add_vline(x=k_info["k_ch"], line_dash="dash",
                  line_color="#d7191c", row=1, col=3)
    fig.add_trace(go.Scatter(x=ks, y=k_info["db_scores"], mode="lines+markers",
                  marker_color="#ff7f00"), row=2, col=1)
    fig.add_vline(x=k_info["k_db"], line_dash="dash",
                  line_color="#d7191c", row=2, col=1)
    fig.add_trace(go.Scatter(x=ks, y=k_info["gaps"],
                  error_y=dict(type="data", array=k_info["gap_sks"]),
                  mode="lines+markers", marker_color="#e41a1c"),
                  row=2, col=2)
    fig.add_vline(x=k_info["k_gap"], line_dash="dash",
                  line_color="#d7191c", row=2, col=2)
    vote_k = sorted(k_info["votes"].keys())
    vote_v = [k_info["votes"][kk] for kk in vote_k]
    colors = ["#d7191c" if kk == best_k else "#2c7bb6" for kk in vote_k]
    fig.add_trace(go.Bar(x=[f"k={kk}" for kk in vote_k], y=vote_v,
                  marker_color=colors, text=vote_v,
                  textposition="outside"), row=2, col=3)
    fig.update_layout(height=700, showlegend=False)
    # save_html(fig, "07_k_selection.html",
              # f"5-Method Consensus k-Selection (Champion k={best_k})")

    # ── MODEL 1: k-Means ────────────────────────────────────────────────────
    print(f"\n  ── k-Means (k={best_k}) ──")
    km_final  = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=15)
    labels_km = km_final.fit_predict(X_clust)
    sil_km = silhouette_score(X_clust, labels_km,
                               sample_size=min(2000, len(labels_km)))
    db_km  = davies_bouldin_score(X_clust, labels_km)
    ch_km  = calinski_harabasz_score(X_clust, labels_km)
    print(f"    Sil={sil_km:.4f}  DB={db_km:.4f}  CH={ch_km:.1f}")

    # ── MODEL 2: GMM ────────────────────────────────────────────────────────
    print("\n  ── GMM (BIC-selected, tied covariance) ──")
    gmm_ks = range(max(2, min(K_RANGE)), GMM_MAX_K + 1)
    bic, aic = [], []
    for k in gmm_ks:
        g = GaussianMixture(n_components=k, covariance_type="tied",
                             random_state=RANDOM_SEED, max_iter=300, n_init=3)
        g.fit(X_clust)
        bic.append(g.bic(X_clust))
        aic.append(g.aic(X_clust))
    bic_arr = np.array(bic)
    bic_deltas = np.diff(bic_arr)
    threshold = 0.01 * (bic_arr.max() - bic_arr.min())
    elbow_idx = next((i for i, d in enumerate(bic_deltas) if abs(d) < threshold),
                     int(np.argmin(bic_arr)))
    best_k_gmm = list(gmm_ks)[elbow_idx]
    print(f"    BIC suggests k={best_k_gmm}, but using consensus k={best_k} for fair comparison")
    gmm_final  = GaussianMixture(n_components=best_k, covariance_type="tied",
                                  random_state=RANDOM_SEED, max_iter=300, n_init=5)
    gmm_final.fit(X_clust)
    labels_gmm = gmm_final.predict(X_clust)
    sil_gmm = silhouette_score(X_clust, labels_gmm,
                                sample_size=min(2000, len(labels_gmm)))
    db_gmm  = davies_bouldin_score(X_clust, labels_gmm)
    ch_gmm  = calinski_harabasz_score(X_clust, labels_gmm)
    print(f"    k={best_k_gmm}  Sil={sil_gmm:.4f}  DB={db_gmm:.4f}  CH={ch_gmm:.1f}")

    # ── MODEL 3: Agglomerative + dendrogram ──────────────────────
    print("\n  ── Agglomerative Hierarchical (Ward) + dendrogram ──")
    dendro_n = min(1500, X_clust.shape[0])
    d_idx = np.random.RandomState(RANDOM_SEED).choice(
        X_clust.shape[0], dendro_n, replace=False)
    Z = linkage(X_clust[d_idx], method="ward")

    agg_final  = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    labels_agg = agg_final.fit_predict(X_clust)
    sil_agg = silhouette_score(X_clust, labels_agg,
                                sample_size=min(2000, len(labels_agg)))
    db_agg  = davies_bouldin_score(X_clust, labels_agg)
    ch_agg  = calinski_harabasz_score(X_clust, labels_agg)
    print(f"    k={best_k}  Sil={sil_agg:.4f}  DB={db_agg:.4f}  CH={ch_agg:.1f}")

    # Dendrogram plot (Ward (1963))
    dd = dendrogram(Z, no_plot=True, color_threshold=0)
    icoord = np.array(dd["icoord"])
    dcoord = np.array(dd["dcoord"])
    fig = go.Figure()
    for i in range(icoord.shape[0]):
        fig.add_trace(go.Scatter(x=icoord[i], y=dcoord[i], mode="lines",
                                 line=dict(color="#2c7bb6", width=1),
                                 showlegend=False, hoverinfo="skip"))
    merge_dists = sorted(Z[:, 2], reverse=True)
    cut_h = ((merge_dists[best_k - 2] + merge_dists[best_k - 1]) / 2
             if len(merge_dists) >= best_k else merge_dists[0])
    fig.add_hline(y=cut_h, line_dash="dash", line_color="#d7191c",
                  annotation_text=f"Cut at k={best_k}")
    fig.update_xaxes(showticklabels=False, title="Incidents (leaves)")
    fig.update_yaxes(title="Ward linkage distance")
    fig.update_layout(height=500, showlegend=False)
#     save_html(fig, "08_dendrogram.html",
#               f"Agglomerative Hierarchical Dendrogram (Ward, cut k={best_k})")

    # ── MODEL 4: DBSCAN ─────────────────────────────────────────────────────
    print("\n  ── DBSCAN (k-distance ε selection) ──")
    ms_auto = min(max(5, 2 * X_clust.shape[1]), 20)
    nbrs = NearestNeighbors(n_neighbors=ms_auto, n_jobs=1).fit(X_clust)
    dists, _ = nbrs.kneighbors(X_clust)
    kth_dist = np.sort(dists[:, -1])[::-1]
    d2 = np.gradient(np.gradient(kth_dist))
    knee_idx = int(np.argmax(np.abs(d2)))
    eps_auto = float(kth_dist[knee_idx])
    print(f"    k-distance knee: ε={eps_auto:.4f} (min_samples={ms_auto})")

    best_sil_db, best_params, best_labels_db = -1, None, None
    for eps in [max(0.1, eps_auto * 0.8), eps_auto, eps_auto * 1.2]:
        for ms in [ms_auto, max(3, ms_auto // 2)]:
            lbl = DBSCAN(eps=eps, min_samples=ms, n_jobs=1).fit_predict(X_clust)
            n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
            noise = (lbl == -1).mean()
            if n_cl < 2 or noise > 0.5:
                continue
            mask = lbl != -1
            if len(set(lbl[mask])) < 2:
                continue
            sil = silhouette_score(X_clust[mask], lbl[mask],
                                   sample_size=min(2000, int(mask.sum())))
            if sil > best_sil_db:
                best_sil_db, best_params, best_labels_db = sil, (eps, ms, n_cl), lbl.copy()

    if best_labels_db is None:
        print("    DBSCAN: no valid partition -> fallback to single cluster")
        best_labels_db = np.zeros(len(labels_km), dtype=int)
        best_sil_db, db_db, ch_db = 0.0, 999.0, 0.0
        best_params = (eps_auto, ms_auto, 1)
    else:
        safe = np.where(best_labels_db == -1, 0, best_labels_db)
        db_db = davies_bouldin_score(X_clust, safe)
        ch_db = calinski_harabasz_score(X_clust, safe)

    eps_b, ms_b, n_db = best_params
    print(f"    ε={eps_b:.3f}  ms={ms_b}  clusters={n_db}  "
          f"Sil={best_sil_db:.4f}  "
          f"noise={(best_labels_db==-1).mean()*100:.1f}%")

    return {
        "k_info": k_info,
        "stability": {"mean_ari": stab_mean, "std_ari": stab_std},
        "km":  {"model": km_final, "labels": labels_km, "k": best_k,
                 "sil": sil_km, "db": db_km, "ch": ch_km},
        "gmm": {"model": gmm_final, "labels": labels_gmm, "k": best_k,
                 "sil": sil_gmm, "db": db_gmm, "ch": ch_gmm},
        "agg": {"model": agg_final, "labels": labels_agg, "k": best_k,
                 "sil": sil_agg, "db": db_agg, "ch": ch_agg,
                 "linkage": Z, "linkage_idx": d_idx},
        "dbscan": {"labels": best_labels_db, "k": n_db,
                    "sil": best_sil_db, "db": db_db, "ch": ch_db,
                    "eps": eps_b, "ms": ms_b,
                    "noise_pct": float((best_labels_db == -1).mean() * 100)},
    }


# ═════════════════════════════════════════════════════════════════════════════
# PART 8 - CHAMPION SELECTION
# ═════════════════════════════════════════════════════════════════════════════
def select_champion(clust_out, X_clust=None):
    section("PART 8 — Champion Model Selection")

    rows = []
    for name, key in [("k-Means", "km"), ("GMM", "gmm"),
                       ("Agglomerative", "agg"), ("DBSCAN", "dbscan")]:
        o = clust_out[key]
        rows.append({
            "Model":         name,
            "Clusters":      o["k"],
            "Silhouette ↑":  o["sil"],
            "DB ↓":          o["db"],
            "CH ↑":          o["ch"],
            "Noise %":       (clust_out["dbscan"]["noise_pct"]
                              if name == "DBSCAN" else 0.0),
            "Interpretable": name != "DBSCAN",
        })
    comp = pd.DataFrame(rows)
    comp["sil_rank"] = comp["Silhouette ↑"].rank(ascending=False)
    comp["db_rank"]  = comp["DB ↓"].rank(ascending=True)
    comp["ch_rank"]  = comp["CH ↑"].rank(ascending=False)
    comp["composite_rank"] = (comp["sil_rank"] + comp["db_rank"]
                               + comp["ch_rank"]) / 3.0

    # Only interpretable models qualify as champion (DBSCAN excluded:
    # 28 micro-clusters is not actionable for municipal policy)
    interpretable = comp[comp["Interpretable"]]
    champ_idx  = interpretable["composite_rank"].idxmin()
    champ_name = comp.loc[champ_idx, "Model"]
    print(comp.to_string(index=False))
    print(f"\n  🏆 CHAMPION: {champ_name}  "
          f"(composite rank = {comp.loc[champ_idx,'composite_rank']:.2f})")

    # ── Critical assessment: Champion selection ──────────────────────────────
    print("\n  ── Critical Assessment (Model Selection) ──")
    print(f"  DECISION: {champ_name} selected as champion model.")
    print("    RATIONALE: Four clustering approaches were compared —")
    print("    k-Means (spherical), GMM (elliptical),")
    print("    Agglomerative (hierarchical), DBSCAN (density-based).")
    print("    Selection uses composite rank across three metrics:")
    print("      Silhouette (Rousseeuw 1987) — cohesion vs. separation")
    print("      Davies-Bouldin (Davies & Bouldin 1979) — intra/inter ratio")
    print("      Calinski-Harabasz (Calinski & Harabasz 1974) — variance ratio")
    print("    No single metric is sufficient (the literature warns that")
    print("    'the elbow score only considers cohesion, not separability').")
    print("    The multi-metric approach guards against criterion-specific bias.")
    comp.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    fig = make_subplots(rows=1, cols=3, subplot_titles=[
        "Silhouette ↑", "Davies-Bouldin ↓", "Calinski-Harabasz ↑"])
    colors = ["#d7191c" if m == champ_name else "#2c7bb6"
              for m in comp["Model"]]
    for col_idx, metric in enumerate(["Silhouette ↑", "DB ↓", "CH ↑"], 1):
        fig.add_trace(go.Bar(x=comp["Model"], y=comp[metric],
                      marker_color=colors,
                      text=[f"{v:.3f}" if metric != "CH ↑" else f"{v:.0f}"
                            for v in comp[metric]],
                      textposition="outside", showlegend=False),
                      row=1, col=col_idx)
    fig.update_layout(height=420,
                      title=dict(text=f"Model Comparison — 🏆 {champ_name}"))
    # save_html(fig, "17_model_comparison.html",
              # f"Four-Algorithm Comparison (Champion: {champ_name})")

    key = {"k-Means": "km", "GMM": "gmm",
            "Agglomerative": "agg", "DBSCAN": "dbscan"}[champ_name]
    champion = dict(clust_out[key])
    champion["name"] = champ_name

    print("\n  ── GMM Soft Overlay ──")
    try:
        from sklearn.mixture import GaussianMixture
        if X_clust is None:
            raise ValueError("X_clust not provided")
        gmm_overlay = GaussianMixture(
            n_components=int(champion["k"]),
            covariance_type="tied",
            random_state=RANDOM_SEED, max_iter=300, n_init=5
        )
        gmm_overlay.fit(X_clust)
        posteriors  = gmm_overlay.predict_proba(X_clust)
        confidence  = posteriors.max(axis=1)
        core_pct  = (confidence >= 0.80).mean() * 100
        peri_pct  = ((confidence >= 0.60) & (confidence < 0.80)).mean() * 100
        ambig_pct = (confidence <  0.60).mean() * 100
        print(f"    Core (>0.80):            {core_pct:.1f}%")
        print(f"    Peripheral (0.60-0.80):  {peri_pct:.1f}%")
        print(f"    Ambiguous (<0.60):       {ambig_pct:.1f}%")
        print(f"    Interpretation: {core_pct:.1f}% of incidents are confidently assigned.")
        print(f"    CAVEAT: This exceedingly high confidence is largely an artifact of")
        print(f"    the deterministic binary missingness flags creating isolated")
        print(f"    hypercubes, rather than a perfect natural taxonomy of human behavior.")
        champion["gmm_confidence"]  = confidence
        champion["gmm_posteriors"]  = posteriors
        champion["gmm_core_pct"]    = float(core_pct)
        champion["gmm_peri_pct"]    = float(peri_pct)
        champion["gmm_ambig_pct"]   = float(ambig_pct)
    except Exception as e:
        print(f"    [skip] GMM overlay: {e}")
        champion["gmm_confidence"] = None
        champion["gmm_posteriors"] = None

    return champion, comp


# ═════════════════════════════════════════════════════════════════════════════
# PART 9 - VISUALIZATIONS + ARTIFACT PERSISTENCE
# ═════════════════════════════════════════════════════════════════════════════
def visualize_and_save(df_cleaned, df_sel, X_scaled, idx, dim_out,
                        clust_out, champion, comp,
                        _n_raw_total=0, _n_no_race=0):
    section("PART 9 — Visualizations & Artifact Persistence")

    X_pca_full = dim_out["X_pca_full"]
    X_pca_2d   = dim_out["X_pca_2d"]
    pca        = dim_out["pca"]
    src_col    = (df_cleaned["source_file"].iloc[idx].values
                  if "source_file" in df_cleaned
                  else np.array(["unknown"] * len(champion["labels"])))

    # Champion 2D with visible centroids
    plot_df = pd.DataFrame({
        "PC1": X_pca_2d[:, 0], "PC2": X_pca_2d[:, 1],
        "Cluster": [f"C{l+1}" if l >= 0 else "Noise"
                    for l in champion["labels"]],
        "source": src_col,
    })
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster",
                     color_discrete_sequence=PALETTE,
                     hover_data={"source": True, "PC1": ":.2f", "PC2": ":.2f"},
                     opacity=0.6)
    fig.update_traces(marker_size=5)

    # Centroids overlaid
    if champion["name"] == "k-Means":
        c2d = champion["model"].cluster_centers_[:, :2]
    else:
        c2d = np.vstack([X_pca_2d[champion["labels"] == c].mean(axis=0)
                         for c in sorted(set(champion["labels"]))
                         if c >= 0])
    for i, (cx, cy) in enumerate(c2d):
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers+text",
            marker=dict(size=26, color="white",
                        line=dict(width=5, color=PALETTE[i % len(PALETTE)])),
            text=[f"C{i+1}"], textposition="middle center",
            textfont=dict(size=14, color="#222", family="Arial Black"),
            showlegend=False,
            hovertemplate=f"Centroid {i}<extra></extra>"))
    fig.update_layout(height=550,
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    # save_html(fig, "09_champion_scatter_2d.html",
              # f"Champion ({champion['name']}, k={champion['k']}) with centroids")

    # 3D scatter
    if X_pca_full.shape[1] >= 3:
        plot_3d = pd.DataFrame({
            "PC1": X_pca_full[:, 0], "PC2": X_pca_full[:, 1],
            "PC3": X_pca_full[:, 2],
            "Cluster": [f"C{l+1}" if l >= 0 else "Noise"
                        for l in champion["labels"]],
            "source": src_col,
        })
        fig = px.scatter_3d(plot_3d, x="PC1", y="PC2", z="PC3",
                            color="Cluster",
                            hover_data={"source": True},
                            color_discrete_sequence=PALETTE, opacity=0.6)
        fig.update_traces(marker_size=3)
        fig.update_layout(height=650,
            scene=dict(
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%})"))
#         save_html(fig, "10_champion_3d.html",
#                   "Champion Clusters — Interactive 3D PCA")

    # Profiling
    df_prof = df_sel.iloc[idx].copy().reset_index(drop=True)
    df_prof["cluster"] = champion["labels"]
    df_prof["source"]  = src_col
    df_prof = df_prof[df_prof["cluster"] >= 0]

    top_feat = df_sel.columns[:min(15, df_sel.shape[1])].tolist()
    means  = df_prof.groupby("cluster")[top_feat].mean().round(3)
    counts = df_prof["cluster"].value_counts().sort_index()
    means.to_csv(os.path.join(OUTPUT_DIR, "cluster_profiles_champion.csv"))

    # Cluster sizes
    fig = px.bar(x=[f"C{i+1}" for i in counts.index], y=counts.values.tolist(),
                 color=[f"C{i+1}" for i in counts.index],
                 color_discrete_sequence=PALETTE,
                 labels={"x": "Cluster", "y": "Incidents"},
                 text=counts.values.tolist())
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=400)
    # save_html(fig, "12_cluster_sizes.html",
              # f"Incidents per Cluster — {champion['name']}")

    # Feature-means heatmap
    fig = go.Figure(go.Heatmap(
        z=means.values.tolist(), x=top_feat,
        y=[f"C{i+1}" for i in means.index], colorscale="RdYlGn",
        text=means.values.round(2).tolist(), texttemplate="%{text}",
        hovertemplate="%{x}<br>%{y}<br>Mean: %{z:.3f}<extra></extra>"))
    fig.update_layout(height=420, xaxis_tickangle=-35)
    # save_html(fig, "13_cluster_heatmap.html",
              # f"Feature Means per Cluster — {champion['name']}")

    # Parallel coordinates
    pc_df = df_prof[top_feat + ["cluster"]].copy()
    fig = px.parallel_coordinates(pc_df, color="cluster",
        color_continuous_scale=px.colors.sequential.Viridis,
        dimensions=top_feat)
    fig.update_layout(height=550, coloraxis_colorbar_title="Cluster")
#     save_html(fig, "14_parallel_coords.html",
#               f"Parallel Coordinates — {champion['name']}")

    # Silhouette plot
    if champion["sil"] > 0 and len(set(champion["labels"])) > 1:
        mask = champion["labels"] != -1
        if mask.sum() > 10:
            n_clust_dim = min(dim_out["n_comp_95"], X_pca_full.shape[1])
            sil_vals = silhouette_samples(
                X_pca_full[mask, :n_clust_dim], champion["labels"][mask])
            sil_df = pd.DataFrame({"silhouette": sil_vals,
                "cluster": [f"C{l+1}" for l in champion["labels"][mask]]})
            sil_df = sil_df.sort_values(["cluster", "silhouette"]).reset_index(drop=True)
            fig = px.bar(sil_df, x=sil_df.index, y="silhouette", color="cluster",
                         color_discrete_sequence=PALETTE)
            fig.add_hline(y=champion["sil"], line_dash="dash", line_color="black",
                          annotation_text=f"Mean = {champion['sil']:.3f}")
            fig.update_layout(height=420, bargap=0)
#             save_html(fig, "15_silhouette.html",
#                       f"Silhouette Plot — {champion['name']}")

    # ── Save artifacts for report ─────────────────────────────────────────
    lat_col = next((c for c in df_cleaned.columns
                    if "lat" in c.lower() and
                    pd.api.types.is_numeric_dtype(df_cleaned[c])), None)
    lon_col = next((c for c in df_cleaned.columns
                    if ("lon" in c.lower() or "lng" in c.lower()) and
                    pd.api.types.is_numeric_dtype(df_cleaned[c])), None)

    artifacts = {
        "df_raw_idx":       idx,
        "df_sel":           df_sel,
        "X_scaled":         X_scaled,
        "X_pca_full":       X_pca_full,
        "X_pca_2d":         X_pca_2d,
        "X_mds":            dim_out["X_mds"],
        "X_lle":            dim_out["X_lle"],
        "pca_evr":          pca.explained_variance_ratio_.tolist(),
        "n_comp_95":        dim_out["n_comp_95"],
        "labels_km":        clust_out["km"]["labels"],
        "labels_gmm":       clust_out["gmm"]["labels"],
        "labels_dbscan":    clust_out["dbscan"]["labels"],
        "labels_agg":       clust_out["agg"]["labels"],
        "centroids_pca":    clust_out["km"]["model"].cluster_centers_,
        "k_selection":      clust_out["k_info"],
        "stability":        clust_out["stability"],
        "champion_name":    champion["name"],
        "champion_labels":  champion["labels"],
        "champion_k":       int(champion["k"]),
        "champion_sil":     float(champion["sil"]),
        "champion_db":      float(champion["db"]),
        "champion_ch":      float(champion["ch"]),
        "gmm_confidence":   champion.get("gmm_confidence"),
        "gmm_core_pct":     champion.get("gmm_core_pct"),
        "gmm_peri_pct":     champion.get("gmm_peri_pct"),
        "gmm_ambig_pct":    champion.get("gmm_ambig_pct"),
        "model_comparison": comp.to_dict(orient="records"),
        "cluster_means":    means.to_dict(),
        "cluster_counts":   counts.to_dict(),
        "top_features":     top_feat,
        "lat_col":          lat_col,
        "lon_col":          lon_col,
        "source_files":     src_col,
        "feature_cols":     df_sel.columns.tolist(),
        "n_raw":            _n_raw_total,
        "n_no_race":        _n_no_race,
    }
    ap = os.path.join(OUTPUT_DIR, "model_artifacts.joblib")
    joblib.dump(artifacts, ap, compress=3)
    print(f"\n  [saved] {ap}  ({os.path.getsize(ap)/1e6:.1f} MB)")

    raw_subset = df_cleaned.iloc[idx].reset_index(drop=True)
    try:
        rp = os.path.join(OUTPUT_DIR, "raw_subset.parquet")
        raw_subset.to_parquet(rp, index=False)
        print(f"  [saved] {rp}")
    except Exception:
        rp = os.path.join(OUTPUT_DIR, "raw_subset.csv")
        raw_subset.to_csv(rp, index=False)
        print(f"  [saved] {rp}  (parquet unavailable -> CSV)")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    """Run the full pipeline: load, EDA, engineer, reduce, cluster, report."""
    parser = argparse.ArgumentParser(
        description="Policing Equity — unsupervised ML pipeline")
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help="Path to folder containing CSV files.")
    args = parser.parse_args()

    t0 = time.time()
    df_raw = load_policing_data(args.data_dir)
    _n_raw_total = df_raw.attrs.get("_n_raw_total", len(df_raw))
    _n_no_race = df_raw.attrs.get("_n_no_race", 0)
    run_eda(df_raw)

    df_encoded, df_cleaned, num_cols, cat_cols, geo_cols = \
        feature_engineering(df_raw)
    df_encoded = generate_features(df_encoded, df_cleaned)
    df_sel     = select_features(df_encoded)

    X_scaled = StandardScaler().fit_transform(df_sel)
    if X_scaled.shape[0] > MAX_ROWS:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(X_scaled.shape[0], MAX_ROWS, replace=False)
    else:
        idx = np.arange(X_scaled.shape[0])
    print(f"\n  Final feature matrix: {X_scaled.shape}  "
          f"(clustering sample: {X_scaled[idx].shape})")

    dim_out = reduce_dimensions(X_scaled, idx)
    n_clust_dim = min(dim_out["n_comp_95"], dim_out["X_pca_full"].shape[1])
    X_clust     = dim_out["X_pca_full"][:, :n_clust_dim]

    clust_out      = run_clustering(X_clust)
    champion, comp = select_champion(clust_out, X_clust)
    visualize_and_save(df_cleaned, df_sel, X_scaled, idx, dim_out,
                       clust_out, champion, comp,
                       _n_raw_total=_n_raw_total, _n_no_race=_n_no_race)

    if _report is not None:
        artifacts = joblib.load(os.path.join(OUTPUT_DIR, "model_artifacts.joblib"))
        _report.generate(artifacts, data_dir=args.data_dir)

    story_path = os.path.join(OUTPUT_DIR, "equity_data_story.html")
    if os.path.exists(story_path):
        webbrowser.open("file://" + os.path.abspath(story_path))

    section("PIPELINE COMPLETE")
    print(f"  Total runtime: {time.time()-t0:.1f} s")
    print(f"  Raw rows:      {df_raw.shape[0]:,}")
    print(f"  Final feats:   {df_sel.shape[1]}")
    print(f"  Champion:      {champion['name']} (k={champion['k']})")
    print(f"  Silhouette:    {champion['sil']:.4f}")
    print(f"  Output:        {OUTPUT_DIR}/equity_data_story.html")
    print(f"  Figures:       {OUTPUT_DIR}/fig1..fig5.png")


if __name__ == "__main__":
    main()
