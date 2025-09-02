# 🧬 Machine Learning Pipeline for Omics Data

## Overview
This repository contains a **generalized machine learning (ML) and deep learning (DL) pipeline** for analyzing **tabular omics datasets**, including proteomics, transcriptomics, metabolomics, and other high-dimensional biomarker data. The pipeline is modular, reproducible, and suitable for binary or multiclass classification problems.

It has been tested on:
- **Hypertension transcriptomics data (ENSAT-HT project)**
- **Alzheimer’s proteomics data (Bio-Hermes dataset)**

---

## ✨ Key Features
- Preprocessing (missing data handling, normalization, scaling)
- Feature selection (statistical filtering, univariate tests, domain-driven panels)
- Multiple ML/DL classifiers:
  - **Random Forest (RF)**
  - **Gradient Boosting (GB, e.g., XGBoost/LightGBM)**
  - **Neural Networks (NN)**
- Evaluation metrics: Accuracy, Sensitivity, Specificity, Balanced Accuracy, PPV, NPV
- Feature importance analysis and consensus biomarker signature discovery
- Robustness checks with multiple train/test splits (60/40, 70/30, 80/20, 90/10)
- Reproducibility ensured with fixed seeds and transparent workflows

---

## 📊 Methods in Detail

### Preprocessing
- Data ingestion from CSV/TSV/Excel.
- Complete-case filtering or fold-specific imputation for missing values.
- Log-transformation and standardization to normalize features.
- Stratified splitting to preserve class balance.

### Feature Screening
- Univariate **t-tests** or **ANOVA** for continuous features.
- χ² tests for categorical features.
- Retention of significant features (e.g., p < 0.05) or top-ranked by effect size.
- Optional biologically curated feature sets (e.g., pathway-specific panels).

### Machine Learning Models
- **Random Forest (RF):**
  - Bootstrap-aggregated decision trees.
  - Handles high-dimensional and correlated features.
  - Feature importance from Gini decrease.

- **Gradient Boosting (GB):**
  - Sequentially optimized trees (e.g., XGBoost, LightGBM).
  - Feature importance based on gain/cover.
  - Strong predictive performance with fewer hyperparameters.

- **Neural Networks (NN):**
  - Feed-forward multilayer perceptron.
  - Nonlinear feature interaction modeling.
  - Weight-based feature influence extraction.
  - Tunable hidden layers, units, and weight decay.

### Evaluation
- Metrics calculated on held-out test sets:
  - **Accuracy**
  - **Sensitivity (Recall)**
  - **Specificity**
  - **Balanced Accuracy**
  - **Positive Predictive Value (PPV)**
  - **Negative Predictive Value (NPV)**
- Confusion matrices and ROC curves generated for each model.
- Feature overlap analysis across models via Venn diagrams.

---

## 📂 Applicable Data Repositories
This pipeline is generalizable to any tabular omics dataset. Applicable repositories include:
- **Gene Expression Omnibus (GEO)** – transcriptomics and proteomics.
- **ArrayExpress** – functional genomics data.
- **dbGaP** – genomic and clinical study data.
- **ADDI (Alzheimer’s Disease Data Initiative)** – Alzheimer’s biomarker datasets, including Bio-Hermes.
- **ENSAT-HT** (European Network for the Study of Adrenal Tumors – Hypertension Transcriptomics).

---

## ✅ Tested Datasets
- **ENSAT-HT transcriptomics (hypertension):** Transcriptome-wide expression profiles used for biomarker discovery and patient stratification.
- **Bio-Hermes proteomics (Alzheimer’s):** Plasma proteomic profiles with amyloid PET/CSF status labels used for binary classification tasks.

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

### Input Format
- **Samples as rows**, features as columns.
- Must include a column with the **outcome label** (binary or multiclass).
- Missing values permitted but will be handled during preprocessing.

### Usage Example
```python
from pipeline import OmicsPipeline

# Initialize pipeline
pipeline = OmicsPipeline(data="data/omics.csv", label="amyloid_status")

# Preprocess data
pipeline.preprocess()

# Train models
pipeline.train_models(models=["RF", "GB", "NN"])

# Evaluate
pipeline.evaluate()

# Extract feature importance
pipeline.feature_signature()
```

---

## 📜 License
MIT License – free to use, modify, and distribute with attribution.
