# Melting Point Prediction with Machine Learning

This repository contains a reproducible **cheminformatics and machine-learning pipeline** for predicting the melting points of organic molecules from molecular structure.  
The project combines **RDKit-based molecular featurisation** with multiple regression models and a stacked ensemble.

It is designed to demonstrate **best practices for chemical data modelling**, with direct relevance to **computer-aided drug discovery (CADD)** and molecular property prediction workflows.

---

## Project focus

The emphasis of this project is on:

- chemically motivated data cleaning and molecule filtering
- robust feature engineering using RDKit descriptors and fingerprints
- careful model selection using cross-validation
- strict avoidance of data leakage
- evaluation on a **single, held-out final test set**

---

## Repository overview

```text
melting-point-ml/
├── data/
│   ├── raw/            # Original, untouched CSV files (not tracked)
│   └── processed/      # Final datasets used for modelling
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_base_model_optimization.ipynb
│   └── 04_ensemble_stacking.ipynb
├── reports/
│   ├── best_params/    # Best hyperparameters from CV (JSON)
│   └── figures/        # Optional saved plots
├── src/
│   └── mp/             # Core Python package (feature engineering + models)
├── pyproject.toml
├── requirements.txt
└── README.md
```
---

## Data sources

The raw melting point data used in this project comes from two publicly shared datasets:

- [**Jean Claude Bradley Open Melting Point Dataset**](https://figshare.com/articles/dataset/Jean_Claude_Bradley_Open_Melting_Point_Datset/1031637)
- [**Jean Claude Bradley Double Plus Good Highly Curated and Validated Melting Point Dataset**](https://figshare.com/articles/dataset/Jean_Claude_Bradley_Double_Plus_Good_Highly_Curated_and_Validated_Melting_Point_Dataset/1031638?file=1503991)

---

## Environment requirements

### RDKit (important)

**`01_data_loading.ipynb`** requires a Python environment with **RDKit installed**, as it performs molecular parsing and descriptor generation directly from SMILES strings.  
All subsequent notebooks operate on pre-generated CSV files and do **not** require RDKit.

Recommended options for running the first notebook:
- Conda / Mamba environment with RDKit
- A chemistry-specific Jupyter kernel (e.g. `rdkit`)

Ensure the RDKit-enabled kernel is selected before running `01_data_loading.ipynb`.

## How to run

### 1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   pip install -e .
```

### 2. Prepare Data

```
data/raw/
├── BradleyMeltingPointDataset.csv
└── BradleyDoublePlusGoodMeltingPointDataset.csv
```

### 3. Notebook execution order
    01_data_loading.ipynb (requires RDKit)
    02_EDA.ipynb
    03_base_model_optimization.ipynb
    04_ensemble_stacking.ipynb

**Metric**: Mean Absolute Error (MAE, °C)
**Data split**: 5% final test set held out once and never used during model selection or stacking.

---

## Data preparation and featurization

### `01_data_loading.ipynb`

This notebook builds the dataset used throughout the project.

**Steps performed:**
1. Load two public melting point datasets (full and curated)
2. Merge and deduplicate entries by SMILES, keeping curated values when available
3. Clean and filter molecules:
   - remove missing or invalid SMILES
   - remove charged species
   - restrict to organic molecules
4. Generate molecular features:
   - RDKit 2D descriptors (physicochemical, ring, shape, connectivity)
   - Morgan fingerprints (radius = 2, 2048 bits)
5. Create a fixed data split:
   - 5% held out as a final test set (never used in training or tuning)
   - 95% used for cross-validation
6. Save processed datasets to `data/processed/`

**Outputs:**
- `full_dataset.csv`
- `train_val.csv`
- `final_test.csv`

---

## Exploratory Data Analysis

### `02_EDA.ipynb`

This notebook performs exploratory analysis on the fully featurized dataset generated in `01_data_loading.ipynb`.  
The goal is to validate that the melting point target and key molecular features are well-behaved and suitable for regression modelling.

**Key analyses:**
- Distribution of the melting point target (`mpC`), including summary statistics and IQR-based outlier analysis
- Visual inspection of structure–property relationships between melting point and selected RDKit descriptors (e.g. molecular weight, TPSA, hydrogen bonding, ring count)
- Correlation analysis:
  - feature–target correlations to identify informative descriptors
  - feature–feature correlations to highlight redundancy and motivate later feature pruning

Morgan fingerprint bits are excluded from the plots for interpretability.  
Figures generated in this notebook can optionally be saved to `reports/figures/` for reference.

---

## Model training and hyperparameter optimisation

### `03_base_model_optimization.ipynb`

This notebook trains and tunes multiple regression models for melting-point prediction using a
consistent, cross-validated workflow.

### Overview

- Models are trained on the pre-processed training/validation dataset generated in earlier notebooks.
- Each model wraps a shared feature-engineering pipeline and exposes a scikit-learn compatible API.
- Hyperparameters are optimised using `RandomizedSearchCV` with mean absolute error (MAE) as the metric.
- A separate hold-out split is used to provide an additional, unbiased performance check for each model.

### Optimised models

The following base learners are tuned independently:

- CatBoost regressor  
- XGBoost regressor  
- k-Nearest Neighbours regressor  
- Feed-forward neural network (TensorFlow/Keras)

Each model defines its own search space, focusing on regions that were empirically found to provide
a good trade-off between performance and training time.

### Outputs

For each model, the notebook records:

- cross-validated MAE
- hold-out MAE
- total optimisation time
- best hyperparameter configuration

The best parameters for each model are exported as JSON files in `reports/best_params/` and are
reused unchanged in the final ensemble and test-set evaluation.

---

## Ensemble stacking and final evaluation

### `04_ensemble_stacking.ipynb`

The final notebook builds a stacked ensemble using the tuned base models.

### Approach

1. Load the best hyperparameters for each base learner from `reports/best_params/`.
2. Generate **out-of-fold (OOF)** predictions for each base model using K-fold cross-validation.
3. Train a **Ridge regression** meta-model on the OOF prediction matrix.
4. Refit base models on the full training set and evaluate on the held-out final test set.

OOF stacking ensures the meta-model is trained only on predictions generated from data not seen by each base model during fitting, preventing leakage.

### Outputs

- Ensemble MAE on the final test set
- Baseline MAE for each individual base model
- Exported ensemble metrics and stacker weights in `reports/`

---

## Reproducibility and ML good practice 

- Random seeds fixed throughout
- Feature engineering performed inside cross-validation folds
- Final test set held out once and never used during:
   - feature selection
   - hyperparameter tuning
   - ensemble training
- Out-of-fold predictions used for stacking

---

## Final results

Model performance on the **held-out final test set** (never used during training or hyperparameter tuning), evaluated using **mean absolute error (MAE, °C)**:

| Model        | Test MAE (°C) |
|--------------|---------------|
| **Ensemble (stacked)** | **23.74** |
| CatBoost     | 24.85 |
| XGBoost      | 24.46 |
| Neural network | 25.26 |
| kNN          | 28.42 |

The stacked ensemble outperforms all individual base models, demonstrating that the models capture complementary structure–property relationships that are effectively combined by the meta-learner.

Further gains are likely possible with extended hyperparameter searches and alternative representations, but the current implementation prioritises methodological correctness and reproducibility.

