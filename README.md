# 🔍 Opening the Black Box
### Telecom Churn Prediction & CVM Analysis

A full end-to-end Customer Value Management (CVM) project built on telecom subscriber data. The project covers churn prediction, customer segmentation, model explainability (XAI), and data-driven campaign design.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Model Explainability (XAI)](#model-explainability-xai)
- [CVM Campaigns](#cvm-campaigns)
- [Results](#results)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Known Issues](#known-issues)
- [Output Files](#output-files)

---

## Overview

This notebook investigates **why customers churn** and **what we can do about it** — not just by building an accurate model, but by opening the black box to understand the *mechanics* behind each prediction.

| Stage | Technique |
|-------|-----------|
| EDA | Correlation analysis, distribution plots, categorical breakdowns |
| Segmentation | K-Means clustering (K=5), PCA visualization |
| Prediction | Logistic Regression, Random Forest, Gradient Boosting |
| Explainability | SHAP, LIME, Anchors, DALEX, DiCE, PDP, ICE, ALE, Surrogate |
| Business Impact | SHIELD & ASCEND campaign design, revenue-at-risk analysis |

---

## Project Structure

```
Opening_the_Black_Box/
│
├── Opening_the_Black_Box_clean.ipynb   # Main notebook
├── README.md                           # This file
│
└── outputs/                            # Generated PNG charts
    ├── 01_churn_overview.png
    ├── 02_feature_correlations.png
    ├── 03_distributions_churn_vs_retained.png
    ├── 04_categorical_churn_rates.png
    ├── 05_elbow_silhouette.png
    ├── 06_segmentation_analysis.png
    ├── 07_pca_segments.png
    ├── 08_model_evaluation.png
    ├── 09_campaign_analysis.png
    ├── 10_risk_tier_analysis.png
    └── 11_strategic_dashboard.png
```

---

## Dataset

**Source:** `cell2celltrain.csv` — publicly available telecom churn dataset (Cell2Cell)

| Property | Value |
|----------|-------|
| Rows | ~51,000 subscribers |
| Columns | 58 raw features |
| Target | `Churn` (Yes / No) |
| Churn Rate | 28.8% |

**Key features used:**
- Usage: `MonthlyMinutes`, `OverageMinutes`, `DroppedCalls`
- Revenue: `MonthlyRevenue`, `TotalRecurringCharge`
- Tenure: `MonthsInService`, `CurrentEquipmentDays`
- Engagement: `CustomerCareCalls`, `RetentionCalls`, `ThreewayCalls`
- Demographics: `AgeHH1`, `IncomeGroup`, `CreditRating`

---

## Pipeline

### 1 · Feature Engineering
- Binary encoding of Yes/No columns (`_Bin` suffix)
- Ordinal encoding of `CreditRating` (1–7 scale)
- Derived features: `EngagementScore`, `DroppedRate`, `RevenuePerMinute`
- Quartile bucketing: `RevQuartile`, `TenureQuartile`, `EquipQuartile`

### 2 · EDA
- Churn rate by revenue, tenure, equipment age, and care calls quartile
- Retention team funnel analysis
- Pearson correlation heatmap with churn
- Distribution overlays: churned vs retained

### 3 · Segmentation (K-Means, K=5)
Elbow method + silhouette scores → K=5 selected.

| # | Segment | Tag | Churn Rate | Avg Revenue |
|---|---------|-----|-----------|-------------|
| 1 | Silent Defectors | CRISIS | ~45% | ~$57 |
| 2 | Standard Base | MAINTAIN | ~27% | ~$45 |
| 3 | Aging Loyalists | UPGRADE | ~31% | ~$44 |
| 4 | Power Users | UPSELL | ~22% | ~$241 |
| 5 | Active Engaged | GROW | ~25% | ~$94 |

### 4 · Churn Prediction
Three models trained with `StandardScaler` + `StratifiedKFold(5)` cross-validation:

| Model | AUC-ROC | F1 | CV-AUC |
|-------|---------|----|--------|
| Logistic Regression | ~0.73 | ~0.47 | ~0.73 |
| Random Forest | ~0.82 | ~0.57 | ~0.81 |
| **Gradient Boosting** | **~0.84** | **~0.59** | **~0.83** |

Best model used for full-dataset churn scoring and risk tier assignment.

**Risk Tiers:**

| Tier | Score Range |
|------|-------------|
| Low | 0.00 – 0.20 |
| Medium | 0.20 – 0.35 |
| High | 0.35 – 0.50 |
| Critical | 0.50 – 1.00 |

---

## Model Explainability (XAI)

The core focus of this project — seven complementary techniques to understand the model:

### Global Methods
| Method | What it answers |
|--------|----------------|
| **Feature Importance** | Which features the model relies on most |
| **Permutation Importance** | Impact of removing each feature (recall-based) |
| **SHAP Summary Plot** | Magnitude + direction of each feature's effect |
| **PDP** | Average effect of a feature on predictions |
| **ALE** | Feature effects corrected for correlations |
| **Global Surrogate** | Simple decision tree that mimics the GB model |

### Local Methods (single prediction)
| Method | What it answers |
|--------|----------------|
| **SHAP Force Plot** | Which features pushed *this* prediction up/down |
| **LIME** | Local linear approximation around *this* customer |
| **Anchors** | IF-THEN rules that lock in *this* prediction |
| **DALEX Break-Down** | Variable contributions in plain language |
| **DiCE Counterfactuals** | "What would need to change for a different outcome?" |
| **ICE** | How *each individual* customer responds to a feature |

---

## CVM Campaigns

### 🛡 Project SHIELD — Retention
- **Target:** Silent Defectors + Aging Loyalists with equipment age > 400 days and churn score ≥ 0.35
- **Goal:** Reduce churn through device upgrade offers and proactive outreach
- **Estimated savings:** ~15% of at-risk monthly revenue

### 🚀 Project ASCEND — Upsell
- **Target:** Power Users + Active Engaged with overage minutes > 50
- **Goal:** Upgrade customers to higher-tier plans before they self-churn from frustration
- **Estimated uplift:** ~$226K annually at 12% conversion rate

---

## Results

```
Total customers         : 51,047
Churn rate              : 28.8%
Monthly revenue at risk : ~$849K

Top churn drivers:
  1. Equipment Age       (r = +0.104)  → older device = more churn
  2. Retention Calls     (r = +0.065)  → already at risk signal
  3. Recurring Charge    (r = -0.061)  → higher plan = less churn
  4. Monthly Minutes     (r = -0.050)  → higher usage = less churn
  5. Handset Models      (r = -0.040)  → more devices = less churn
```

---

## Installation

```bash
git clone https://github.com/<your-username>/opening-the-black-box.git
cd opening-the-black-box

pip install -r requirements.txt
jupyter notebook Opening_the_Black_Box_clean.ipynb
```

Update the data path in **Section 1.2** before running:
```python
DATA_PATH  = r"path/to/cell2celltrain.csv"
OUTPUT_DIR = r"path/to/output/folder"
```

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
shap
lime
anchor-exp          # for Anchors (see note below)
dalex
plotly
dice-ml
pyALE
```

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap lime anchor-exp dalex plotly dice-ml pyALE
```

---

## Known Issues

### ⚠️ `alibi` — numpy binary incompatibility
`alibi` (the original Anchors library) may raise a `ValueError: numpy.dtype size changed` error on Python 3.11+ due to a binary incompatibility between `thinc` and newer numpy versions.

**Fix — use `anchor-exp` instead (recommended):**
```bash
pip install anchor-exp
```
```python
from anchor import anchor_tabular

explainer = anchor_tabular.AnchorTabularExplainer(
    class_names=['Stayed', 'Churn'],
    feature_names=MODEL_FEATURES,
    train_data=X_train.values
)
exp = explainer.explain_instance(
    X_test.iloc[8].values,
    gb_model.predict,
    threshold=0.95
)
print("Rule      :", " AND ".join(exp.names()))
print("Precision :", round(exp.precision(), 2))
print("Coverage  :", round(exp.coverage(), 2))
```

**Alternative fix — downgrade numpy:**
```bash
pip install "numpy<2.0" --force-reinstall
pip install alibi
```
Then restart the Jupyter kernel.

---

## Output Files

The notebook saves 11 publication-ready PNG charts to `OUTPUT_DIR`:

| File | Contents |
|------|----------|
| `01_churn_overview.png` | Churn split, quartile breakdowns, retention funnel |
| `02_feature_correlations.png` | Pearson correlation with churn — all numeric features |
| `03_distributions_churn_vs_retained.png` | Overlaid histograms for key variables |
| `04_categorical_churn_rates.png` | Churn rate by categorical & binary variables |
| `05_elbow_silhouette.png` | K-Means elbow + silhouette for K=2..10 |
| `06_segmentation_analysis.png` | Scatter, radar, donut, revenue-at-risk by segment |
| `07_pca_segments.png` | PCA 2D projection of 5 customer segments |
| `08_model_evaluation.png` | ROC, PR curves, confusion matrix, feature importance |
| `09_campaign_analysis.png` | SHIELD & ASCEND targeting + ROI waterfall |
| `10_risk_tier_analysis.png` | Risk tier distribution, churn rates, calibration |
| `11_strategic_dashboard.png` | Executive KPI cards, segment matrix, 12-month roadmap |

---

<p align="center">
  Built with Python · scikit-learn · SHAP · LIME · DALEX · DiCE
</p>
