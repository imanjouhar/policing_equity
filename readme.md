# Policing Equity — Unsupervised Learning Analysis

An unsupervised machine learning pipeline that analyses 1,451,318 policing
records from the [Center for Policing Equity](https://www.kaggle.com/center-for-policing-equity/data-science-for-good)
dataset, identifying structural patterns in enforcement activity across
Minneapolis, Los Angeles, Dallas, Austin, and Indianapolis.

Developed as the Task 2 case study for DLBDSMLUSL01 (Machine Learning:
Unsupervised Learning and Feature Engineering) at IU International University
of Applied Sciences.

## Objective

Policing data has grown too large for manual inspection. This project reduces
the complexity of 109 raw columns across 12 policing event CSV files into an
interpretable set of clusters, revealing which communities experience
disproportionate enforcement — and where officer discretion widens or narrows
those gaps.

## Known Data Quality Issues

The CPE Kaggle dataset contains two data quality issues that the pipeline
handles automatically:

1. **Indianapolis column typo (dept 23-00089):** The use-of-force file
   `23-00089_UOF-P.csv` contains a column named `SUBJECT_RACT` instead
   of `SUBJECT_RACE`. Without correction, all 10,274 records lose their
   subject race data silently. The pipeline detects and renames this
   column before race normalisation (see `feature_engineering()`,
   line 413).

2. **Duplicate department files (49-0009 / 49-00009):** Two department
   folders contain identical CSV files under inconsistently zero-padded
   codes. The pipeline's MD5 hash deduplication (`load_policing_data()`,
   line 270) catches this automatically, retaining only one copy.

These fixes ensure reproducibility. Running the pipeline on the raw
Kaggle download produces the same 1,451,318 post-cleaning records
without manual intervention.

## Methodology

The pipeline follows a structured workflow from raw data to a data story:

**Data preparation** — 12 CSV files are loaded with MD5 deduplication and
schema normalisation. The full dataset also includes American Community Survey
(ACS) Census tables for each department (poverty, income, employment,
education, housing), which are inventoried but not joined to the policing
records because the incident files carry no census tract identifier. Columns
with more than 80% missing values are dropped as structurally incomplete (91
columns removed). A further 10 columns above 25% missing are dropped; 4
columns in the 5–25% band are retained with binary missingness flags.
Demographic columns (race, gender, age) are protected from dropping to
preserve them for equity profiling. Extreme outliers are removed via IQR
fencing at 3×IQR, followed by Winsorisation at P1/P99. Skewed distributions
are log-transformed and all features are standardised to zero mean and unit
variance.

**Feature engineering** — Categorical variables are encoded via one-hot and
frequency encoding. Equity-relevant ratio features are generated (e.g.
force-per-arrest rate, racial arrest ratio). Low-variance features and highly
correlated pairs (|r| > 0.95) are pruned, reaching 11 final features.

**Dimensionality reduction** — Three techniques are compared: PCA, MDS, and
LLE. PCA is selected as the primary method, compressing 11 features into 7
components that retain 95.1% of total variance (36.4% dimensionality
reduction). MDS is rejected due to high stress (533,961); LLE produces
near-zero reconstruction error but fragments the data along categorical
boundaries due to manifold collapse caused by sparse binary features.

**Clustering** — Four algorithms are fitted and compared on the 7-component
PCA space: k-Means, Gaussian Mixture Model (GMM), Agglomerative Hierarchical
(Ward linkage), and DBSCAN. The number of clusters (k = 7) is determined by
consensus across five selection methods (Kneedle elbow, Silhouette peak,
Calinski-Harabasz, Davies-Bouldin, Gap Statistic; 4/5 vote for k = 7).
k-Means is selected as champion based on composite rank across three validity
metrics among interpretable models (Silhouette = 0.7010, DB = 0.4932,
CH = 5,035). DBSCAN achieves higher metric scores (Sil = 0.9344) but
over-segments the binary feature space into 28 micro-clusters, making it
unsuitable for actionable policy.

**Validation** — A GMM soft-membership overlay confirms that 100.0% of
incidents carry posterior confidence above 0.80. Bootstrap resampling yields
ARI = 0.969 ± 0.093, indicating stable partition structure.

## Key Findings

- Black residents appear in Minneapolis policing records at 1.84× their
  Census population share; White residents at 0.74×.
- Force severity varies by race even when the stated reason for force is held
  constant — a pattern that cannot be explained by incident type alone.
- In Los Angeles, officer discretion compresses racial disparity: the OLS
  slope of 0.76 (p = 0.001, R² = 0.978) means that for every 1.0× of
  non-discretionary disparity, discretionary disparity rises by only 0.76×.
- In Dallas, White officers use force on Black subjects 57.6% of the time;
  Black officers do so 58.9% of the time — virtually identical, pointing to
  structural deployment patterns rather than individual bias.
- Use-of-force disparity was stable at 3.3× across three presidencies
  (Bush, Obama, Trump), suggesting institutional rather than political drivers.
- 415,278 records (29%) have no race data recorded — an accountability gap
  the algorithm detected independently.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

The pipeline runs in approximately 2 minutes. When finished, the interactive
report opens automatically in the browser.

## Repository Structure

```
├── main.py              # Full ML pipeline (single entry point)
├── report.py            # Interactive HTML report generator
├── requirements.txt     # Python dependencies
├── README.md
├── data/                # CPE dataset (not tracked in git)
│   └── archive (11)/    # Policing CSVs + ACS Census subfolders
└── outputs/             # Generated by pipeline
```

## Outputs

All outputs are written to the `outputs/` directory:

| File | Description |
|------|-------------|
| `equity_data_story.html` | Interactive 11-part scrollable report with 13 Plotly charts |
| `model_artifacts.joblib` | Serialised pipeline for report generation |
| `raw_subset.csv` | Labelled data subset (5,000 sampled records) for cluster profiling |
| `model_comparison.csv` | Four-algorithm validity metrics |
| `dr_comparison.csv` | Dimensionality reduction method comparison |
| `cluster_profiles_champion.csv` | Feature means per cluster |

## Dataset

Center for Policing Equity (2016). *Data Science for Good — Center for
Policing Equity*. Kaggle.
https://www.kaggle.com/center-for-policing-equity/data-science-for-good

Place the extracted dataset in a `data/` directory at the project root,
or specify a custom path: `python main.py --data-dir /path/to/data`.

## Author

Iman Jouhar — April 2026
