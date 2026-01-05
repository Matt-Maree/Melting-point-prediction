# Melting Point Prediction of Organic Compounds

This repository contains my solution to the Kaggle competition  
**Thermophysical Property: Melting Point Prediction**  
https://www.kaggle.com/competitions/melting-point/overview

The goal of the project is to predict the melting points of organic compounds from their molecular structure, provided as **SMILES strings**, using machine-learning models and RDKit-derived molecular features.

---

## Project Overview

The workflow consists of:

1. Data loading and cleaning  
2. Feature engineering using RDKit  
3. Model training and hyperparameter optimisation  
4. Model ensembling via stacking  
5. Evaluation on an unseen test set  

---

## Notebooks and Structure

### `Mol_features.ipynb`

- Loads multiple datasets containing:
  - SMILES strings
  - Experimental melting point values
- Cleans and standardises the data
- Generates molecular features using **RDKit**, including:
  - Physicochemical descriptors
  - Structural descriptors
- Computes **Morgan fingerprints** and appends them to the feature table

---

### `New_model.ipynb`

This notebook contains the full modelling pipeline, implemented using custom classes.

#### Feature Engineering

- A `FeatureEngineer` class that:
  - Adds additional derived features based on RDKit descriptors
  - Removes highly correlated RDKit features using a configurable correlation threshold
  - Filters Morgan fingerprint bits that are:
    - Too rare
    - Too common  
  based on user-defined frequency thresholds

#### Models

The following regression models are implemented and trained:

- CatBoost
- XGBoost
- K-Nearest Neighbours
- Neural Network (custom implementation)

For each model:
- Hyperparameters are optimised using **RandomizedSearchCV**
- Cross-validation is used to select the best configuration

#### Ensembling

- The best individual models are combined using **Ridge regression stacking**
- The stacked ensemble is used to generate final predictions on the unseen test set

---

## Results

- **Final test MAE:** **24.07**
- Evaluation performed on the Kaggle competition’s unseen test data

---

## Tools and Libraries

- Python
- RDKit
- scikit-learn
- CatBoost
- XGBoost
- NumPy
- Pandas
