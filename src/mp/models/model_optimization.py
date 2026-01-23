from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import make_scorer, mean_absolute_error


def run_random_search_cv(
    df: pd.DataFrame,
    target_col: str,
    cv: int,
    n_iter: int,
    estimator: BaseEstimator,
    param_distributions: Mapping[str, Any],
    n_jobs: int,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: int = 1,
) -> tuple[RandomizedSearchCV, float]:
    """Run RandomizedSearchCV with a fixed holdout set for a quick reality check.

    Notes:
    - The CV score is computed on X_train only.
    - The holdout MAE is computed once using the best CV-selected estimator.
    """
    # 1) Split features/target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2) Holdout split (never seen during CV fitting)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3) CV strategy + scoring (negative MAE so "higher is better" for sklearn)
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # 4) Randomized search
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=mae_scorer,
        cv=cv_strategy,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    # 5) Evaluate best estimator on the holdout set
    y_pred_holdout = search.best_estimator_.predict(X_holdout)
    holdout_mae = mean_absolute_error(y_holdout, y_pred_holdout)

    return search, holdout_mae