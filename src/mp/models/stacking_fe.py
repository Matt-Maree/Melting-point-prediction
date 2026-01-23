from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class StackingResult:
    model_names: List[str]
    oof_meta: np.ndarray                 # shape (n_samples, n_models)
    stacker: Pipeline
    fitted_base_models: Dict[str, BaseEstimator]


def _fit_one_model(
    name: str,
    model: BaseEstimator,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    nn_name: str,
) -> BaseEstimator:
    """
    Fit a cloned model. If it's the NN model, pass explicit validation data
    """
    m = clone(model)

    if name == nn_name:
        
        try:
            m.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        except TypeError:
            m.fit(X_tr, y_tr)
    else:
        m.fit(X_tr, y_tr)

    return m


def build_oof_meta_features(
    X: pd.DataFrame,
    y: np.ndarray,
    base_models: Dict[str, BaseEstimator],
    n_splits: int = 3,
    random_state: int = 42,
    nn_name: str = "nn",
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build out-of-fold predictions matrix for stacking.

    Returns
    -------
    oof_meta : np.ndarray
        OOF prediction matrix of shape (n_samples, n_models)
    model_names : list[str]
        The model order used for columns in oof_meta
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame for reliable column alignment.")

    model_names = list(base_models.keys())
    n_models = len(model_names)
    n_samples = X.shape[0]

    oof_meta = np.zeros((n_samples, n_models), dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), start=1):
        if verbose:
            print(f"\n=== Fold {fold}/{n_splits} ===")

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        for j, name in enumerate(model_names):
            if verbose:
                print(f"  Training '{name}'...")

            fitted = _fit_one_model(
                name=name,
                model=base_models[name],
                X_tr=X_tr,
                y_tr=y_tr,
                X_val=X_val,
                y_val=y_val,
                nn_name=nn_name,
            )

            y_val_pred = fitted.predict(X_val)
            oof_meta[val_idx, j] = y_val_pred

            if verbose:
                fold_mae = mean_absolute_error(y_val, y_val_pred)
                print(f"    {name} fold MAE: {fold_mae:.3f}")

    return oof_meta, model_names


def fit_stacker(
    X_meta: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """
    Fit a simple Ridge stacker with standardization.
    """
    stacker = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha, random_state=random_state),
    )
    stacker.fit(X_meta, y)
    return stacker


def fit_base_models_full(
    X: pd.DataFrame,
    y: np.ndarray,
    base_models: Dict[str, BaseEstimator],
    nn_name: str = "nn",
    verbose: bool = True,
) -> Dict[str, BaseEstimator]:
    """
    Fit cloned base models on full training data.
    """
    fitted_base_models: Dict[str, BaseEstimator] = {}

    for name, template_model in base_models.items():
        if verbose:
            print(f"\nFitting '{name}' on full training data...")

        m = clone(template_model)

        m.fit(X, y)
        fitted_base_models[name] = m

    return fitted_base_models


def build_meta_features(
    X: pd.DataFrame,
    fitted_base_models: Dict[str, BaseEstimator],
    model_names: List[str],
) -> np.ndarray:
    """
    Build meta-features from fitted base models in a fixed order.
    """
    n = X.shape[0]
    n_models = len(model_names)
    meta = np.zeros((n, n_models), dtype=float)

    for j, name in enumerate(model_names):
        meta[:, j] = fitted_base_models[name].predict(X)

    return meta


def evaluate_ensemble(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Return MAE.
    """
    return mean_absolute_error(y_true, y_pred)