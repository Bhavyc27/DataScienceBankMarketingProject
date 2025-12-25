# Bank Intersection (Marketing) — Data Science Final Project

This repository contains an end-to-end **EDA + classification modeling** workflow built on a merged “bank marketing / bank intersection” style dataset (CSV).  
The work was originally developed as a Jupyter notebook and has been **converted into a runnable Python script**.

## What the project does

1. **Loads the dataset** from a CSV file.
2. **Exploratory Data Analysis (EDA)**
   - Dataset shape, dtypes, sample rows
   - Missing-value audit
   - Summary statistics (numerical)
   - Target distribution and a cleaned binary label (`y_binary`)
   - Visualizations: histograms, boxplots (outliers), correlation heatmap
   - Target vs categorical feature plots
3. **Preprocessing**
   - Splits data into train/test (stratified)
   - Standardizes numeric features
   - One-hot encodes categorical features
   - Builds *two* preprocessors:
     - **Sparse** (for linear models that can consume sparse matrices)
     - **Dense** (for tree/boosting models that prefer dense arrays)
4. **Modeling & comparisons**
   - Trains multiple baseline models (classification; plus one regression-as-classification baseline)
   - Compares **dimensionality reduction** techniques:
     - **None** (baseline)
     - **PCA** (dense)
     - **JL** (Johnson–Lindenstrauss using SparseRandomProjection)
     - **TruncatedSVD** (sparse-friendly alternative to PCA)
   - Runs with:
     - **No sampling**
     - **SMOTE** (to address class imbalance)
5. **Coreset / Subsampling experiment**
   - Trains selected models using only a fraction (default 10%) of training data
   - Compares accuracy vs the full-data runs (useful for “scaling” discussion)

## Files

- `bankintersection_project.py` — Python script (converted from the notebook)
- `EDA_prep_bankintersection_final_training (1).ipynb` — original notebook (reference)

> The script preserves the notebook’s structure and results tables, but is organized under a `main()` entry point and supports CLI arguments.

## Requirements

Python 3.9+ recommended.

Install dependencies:

```bash
pip install -U pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy ipython
```

## Dataset expectations

The script expects a CSV that contains at minimum:

- Target column: `y` (values like `yes/no`), which is converted to:
  - `y_binary` (0/1)
- For the modeling section (explicit list used in the notebook/script):

**Numeric features**
- `age`, `campaign`, `duration`, `pdays`, `previous`

**Categorical features**
- `contact`, `default`, `education`, `housing`, `job`, `loan`, `marital`, `month`, `poutcome`

If your file uses different column names, edit the feature lists inside the script (search for `numeric_features = [...]`).

## How to run

### 1) Put your CSV in the same folder as the script (simplest)
```bash
python bankintersection_project.py --data "bank_join_common (1).csv"
```

### 2) Provide an absolute/relative path to the CSV
```bash
python bankintersection_project.py --data "/path/to/your/bank_join_common.csv"
```

### 3) Run without plots (useful on servers / faster runs)
```bash
python bankintersection_project.py --data "bank_join_common (1).csv" --no-plots
```

## What you’ll see (outputs)

The script prints and/or displays:

- Dataset preview (`head()`)
- Missing values summary
- Target distribution (`y_binary` counts)
- Multiple plots (unless `--no-plots` is set)
- Results tables comparing:
  - model type
  - reduction method (None/PCA/JL/SVD)
  - training time
  - test accuracy
  - AUC (when probabilities are available)

It also generates bar plots comparing:
- Accuracy by model/reduction
- Training time by model/reduction
- “Speedup” comparisons vs baseline
- Full vs PCA vs JL vs Coreset comparisons

## Notes on methodology (what to explain in your final viva/report)

### Why two preprocessors?
- Linear models (LogReg/SGD/LinearSVM) work efficiently with **sparse one-hot encoded matrices**.
- Tree/boosting models often run better on **dense arrays**, so the script builds a dense preprocessor too.

### Why SMOTE?
If the positive class (e.g., `y = yes`) is underrepresented, SMOTE synthetically oversamples the minority class in the training set to reduce imbalance-related bias.

### Why PCA vs TruncatedSVD vs JL?
- **PCA**: classic variance-preserving reduction, requires dense matrices (unless using sparse PCA variants).
- **TruncatedSVD**: PCA-like reduction that works with sparse matrices (very useful after one-hot encoding).
- **JL (Random Projections)**: reduces dimension while approximately preserving pairwise distances; extremely scalable.

### What is the coreset experiment showing?
It demonstrates that training on a carefully chosen subset (here: random subsample) can reduce training time, but may lower accuracy—useful for “scaling” discussions.

## Troubleshooting

- **File not found**: pass the correct `--data` path.
- **Column not found**: your CSV column names differ; update feature lists and/or the target column name.
- **`AUC=None`**: some models don’t expose reliable probabilities (`predict_proba`). That’s normal.
- **Long runtime**: use `--no-plots` and/or reduce model complexity (e.g., fewer trees in RandomForest).

## Reproducibility tips

- The script uses fixed `random_state=42` where applicable (train/test split, SMOTE, projections, some models).
- For consistent plots, run with the same library versions.
