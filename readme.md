# Policing Equity — Unsupervised Learning & Feature Engineering

A reproducible machine learning pipeline that applies unsupervised clustering
and feature engineering to 1,451,318 policing records from the
[Center for Policing Equity](https://www.kaggle.com/center-for-policing-equity/data-science-for-good)
dataset, spanning twelve law enforcement departments across nine US states.

The pipeline loads raw CSV files, engineers and selects features, compares
three dimensionality reduction techniques and four clustering algorithms,
and generates a self-contained interactive HTML report with geographic
dot maps, disparity indices, and socioeconomic context — all from a single
command.

Developed as the Task 2 case study for DLBDSMLUSL01 (Machine Learning:
Unsupervised Learning and Feature Engineering) at IU International University
of Applied Sciences.

---

## Objective

Policing records across the United States are collected by individual
departments in inconsistent formats, with varying levels of completeness.
Manually reviewing 1.45 million incidents is not feasible. This project
reduces that complexity into an interpretable set of clusters, answering
three questions:

1. **Do policing outcomes differ by race?** The pipeline computes disparity
   indices comparing each racial group's share of policing encounters to
   their Census population share.
2. **Is the disparity structural or individual?** By comparing officer race
   against subject race (Dallas) and discretionary vs non-discretionary
   arrests (Los Angeles), the analysis separates institutional patterns
   from individual behaviour.
3. **Where does disparity concentrate geographically?** Interactive dot maps
   overlay incident coordinates by subject race for Minneapolis and
   Los Angeles, revealing spatial segregation in enforcement activity.

## Departments Covered

| Department | County / State | Records | Type |
|------------|---------------|---------|------|
| Minneapolis | Hennepin County, MN | 736,273 | UoF + Vehicle Stops |
| St. Paul | Ramsey County, MN | ~86,000 | Vehicle Stops |
| Los Angeles | Los Angeles County, CA | ~541,000 | UoF + Arrests |
| San Francisco | San Francisco County, CA | 394,235 | Incident Reports |
| Boston | Suffolk County, MA | 152,230 | Field Interviews |
| Indianapolis | Marion County, IN | 10,274 | Use of Force |
| Dallas | Dallas County, TX | ~12,000 | Use of Force |
| Austin | Travis County, TX | 131 | UoF / OIS |
| Charlotte | Mecklenburg County, NC | TBD | Use of Force |
| Orlando area | Orange County, FL | TBD | Use of Force |
| Seattle area | King County, WA | TBD | Use of Force |
| Alameda County | Alameda County, CA | TBD | Incident Reports |

All departments contribute to the clustering. In-depth equity analyses
focus on three cities with sufficient data quality and volume:
Minneapolis (vehicle stops and use of force), Los Angeles (arrests and
discretion effect), and Dallas (officer race).

## Methodology

### Data Preparation

Twelve policing event CSV files are loaded with MD5 hash deduplication and
automatic schema normalisation (the Kaggle metadata row is detected and
skipped). The dataset also includes American Community Survey (ACS) Census
tables for each department covering poverty, income, employment, education,
and housing at census-tract level. Columns with more than 80% missing values
are dropped as structurally incomplete (91 columns removed). A further 10
columns above 25% missing are dropped; 4 columns in the 5–25% band are
retained with binary missingness flags. Demographic columns (race, gender,
age) are protected from dropping. Extreme outliers are removed via IQR
fencing at 3×IQR, followed by Winsorisation at P1/P99.

### Feature Engineering

Categorical variables are encoded via one-hot (3–15 unique values) and
frequency encoding (>15 unique values). Equity-relevant ratio features are
generated where matching columns exist. Low-variance features (threshold
0.01) and highly correlated pairs (|r| > 0.95) are pruned, yielding 11
final features.

### Dimensionality Reduction

Three techniques are compared: PCA (linear, global), MDS (metric, global),
and LLE (non-linear, local). PCA is selected as the primary method,
compressing 11 features into 7 components retaining 95.1% of total variance.
MDS is rejected due to high stress (533,961). LLE produces near-zero
reconstruction error but fragments the data along categorical boundaries.

### Clustering

Four algorithms are fitted on the 7-component PCA space:

- **k-Means** — spherical clusters, selected as champion
- **Gaussian Mixture Model** — elliptical clusters, tied covariance
- **Agglomerative Hierarchical** — Ward linkage with dendrogram
- **DBSCAN** — density-based, automatic epsilon via k-distance knee

The number of clusters (k = 7) is determined by consensus across five
independent selection methods (Kneedle elbow, Silhouette, Calinski-Harabasz,
Davies-Bouldin, Gap Statistic; 4 of 5 vote for k = 7). k-Means is selected
as champion based on composite rank across Silhouette (0.7010),
Davies-Bouldin (0.4932), and Calinski-Harabasz (5,035).

### Validation

A GMM soft-membership overlay confirms that 100% of incidents carry
posterior confidence above 0.80. Bootstrap resampling (10 iterations, 80%
subsample) yields ARI = 0.969 ± 0.093, indicating stable partition
structure.

## Key Findings

- **Racial disparity is measurable and consistent.** Black residents appear
  in Minneapolis policing records at 1.84× their Census population share.
  In Los Angeles, Black arrest disparity peaks at 3.8×.

- **Force severity varies by race after controlling for reason.** Even when
  the stated reason for force is held constant, average severity differs
  across racial groups — a pattern incident type alone cannot explain.

- **Officer discretion compresses disparity.** In Los Angeles, the OLS slope
  of 0.76 (p = 0.001, R² = 0.978) means discretionary arrests narrow
  the racial gap compared to non-discretionary arrests.

- **Officer race does not change outcomes.** In Dallas, White officers use
  force on Black subjects 57.6% of the time; Black officers do so 58.9% —
  virtually identical, pointing to structural deployment rather than
  individual bias.

- **Disparity is temporally stable.** Use-of-force disparity held at 3.3×
  across three presidencies (Bush, Obama, Trump), suggesting institutional
  rather than political drivers.

- **Geographic concentration mirrors poverty.** Interactive dot maps reveal
  that Black policing encounters cluster in the same neighbourhoods where
  ACS census data shows the highest poverty rates and lowest incomes.

- **Missing data is structural, not random.** 415,278 records (29%) lack
  race data — the algorithm independently isolated this as a distinct
  cluster, detecting an accountability gap without being instructed to
  look for one.

## Known Data Quality Issues

The CPE Kaggle dataset contains two data quality issues that the pipeline
handles automatically:

1. **Indianapolis column typo (dept 23-00089).** The use-of-force file
   `23-00089_UOF-P.csv` contains a column named `SUBJECT_RACT` instead
   of `SUBJECT_RACE`. Without correction, all 10,274 records lose their
   subject race data silently. The pipeline detects and renames this
   column before race normalisation (see `feature_engineering()`,
   line 413).

2. **Duplicate department files (49-0009 / 49-00009).** Two department
   folders contain identical CSV files under inconsistently zero-padded
   codes. The pipeline's MD5 hash deduplication (`load_policing_data()`)
   catches this automatically, retaining only one copy.

These fixes ensure reproducibility. Running the pipeline on the raw
Kaggle download produces the same 1,451,318 post-cleaning records
without manual intervention.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

The pipeline completes in approximately 90–110 seconds. When finished,
the interactive report opens automatically in the default browser.

To specify a custom data directory:
```bash
python main.py --data-dir /path/to/data
```

## Repository Structure

```
├── main.py                # Full ML pipeline (single entry point)
├── report.py              # Interactive HTML report generator
├── requirements.txt       # Python dependencies
├── README.md
├── data/                  # CPE dataset (not tracked in git)
│   └── archive/           # Dept folders with policing CSVs,
│       ├── Dept_24-00013/ #   ACS Census subfolders, and
│       ├── Dept_49-00033/ #   shapefiles per department
│       └── ...
└── outputs/               # Generated by pipeline
```

## Outputs

All outputs are written to the `outputs/` directory:

| File | Description |
|------|-------------|
| `equity_data_story.html` | Interactive 13-part scrollable report with Plotly charts |
| `map_minneapolis.html` | Geographic dot map of Minneapolis UoF incidents by race |
| `map_los_angeles.html` | Geographic dot map of Los Angeles arrests by race |
| `model_artifacts.joblib` | Serialised pipeline artifacts for report generation |
| `raw_subset.csv` | Labelled data subset (5,000 sampled records) for cluster profiling |
| `model_comparison.csv` | Four-algorithm validity metrics |
| `dr_comparison.csv` | Dimensionality reduction method comparison |
| `cluster_profiles_champion.csv` | Feature means per cluster |

## Dataset

Center for Policing Equity (2016). *Data Science for Good — Center for
Policing Equity*. Kaggle.
https://www.kaggle.com/center-for-policing-equity/data-science-for-good

Place the extracted dataset in a `data/` directory at the project root,
or specify a custom path with `--data-dir`.

## Author

Iman Jouhar — DLBDSMLUSL01 Task 2 — April 2026