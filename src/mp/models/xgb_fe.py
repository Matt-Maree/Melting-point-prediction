import pandas as pd

from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error

from mp.feature_engineer import FeatureEngineer


class XGBFEModel(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        # XGBoost params (tunable)
        learning_rate: float = 0.0635,
        n_estimators: int = 2174,
        max_depth: int = 8,
        min_child_weight: int = 11,
        subsample: float = 0.894,
        colsample_bytree: float = 0.888,
        reg_lambda: float = 6.42,
        reg_alpha: float = 0.928,
        gamma: float = 0.149,
        colsample_bylevel: float = 0.972,
        tree_method: str = "hist",
        max_bin: int = 256,
        objective: str = "reg:absoluteerror",
        eval_metric: str = "mae",
        random_state: int = 42,
        xgb_verbose: bool = False,

        # FeatureEngineer hyperparams (tunable)
        corr_threshold: float = 0.90,
        min_fp_freq: float = 0.0104,
        max_fp_freq: float = 0.913,
        fe_verbose: bool = False,
        target_col: str = "mpC",

        # Optional FE behaviour
        normalize: bool = False,
        onehot_cats: bool = False,
        cat_cols: list | None = None,
        smiles_col: str = "smiles",
        fp_prefix: str = "FP",
    ):
        # --- XGB params ---
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.tree_method = tree_method
        self.max_bin = max_bin
        self.objective = objective
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.xgb_verbose = xgb_verbose

        # --- FE params ---
        self.corr_threshold = corr_threshold
        self.min_fp_freq = min_fp_freq
        self.max_fp_freq = max_fp_freq
        self.fe_verbose = fe_verbose
        self.target_col = target_col

        # --- Optional FE behaviour ---
        self.normalize = normalize
        self.onehot_cats = onehot_cats
        self.cat_cols = cat_cols
        self.smiles_col = smiles_col
        self.fp_prefix = fp_prefix

        # learned in fit()
        self.fe_ = None
        self.model_ = None
        self.cat_feature_indices_ = None
        self.cat_cols_ = []
        self.feature_names_in_ = None

    def fit(self, X, y):
        """
        X: pandas DataFrame (raw features; should NOT include the target column)
        y: target (Series/array)
        """
        # --- ensure DataFrame ---
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Ensure alignment and 1D target
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        self.feature_names_in_ = list(X.columns)

        # 1) build training df with target
        # Attach target so FeatureEngineer can safely exclude it from feature-selection logic
        df_with_target = X.copy()
        df_with_target[self.target_col] = y

        # 2) set up FeatureEngineer with current hyperparams
        # Fit FeatureEngineer on training data only (prevents leakage across CV folds)
        self.fe_ = FeatureEngineer(
            corr_threshold=self.corr_threshold,
            min_fp_freq=self.min_fp_freq,
            max_fp_freq=self.max_fp_freq,
            verbose=self.fe_verbose,
            normalize=self.normalize,
            onehot_cats=self.onehot_cats,
            cat_cols=self.cat_cols,
            target_col=self.target_col,
            smiles_col=self.smiles_col,
            fp_prefix=self.fp_prefix,
        )

        # 3) fit FE and transform training data
        self.fe_.fit(df_with_target)
        X_trans = self.fe_.transform(X)

        # 4) categorical handling 
        if self.onehot_cats:
            cat_cols = []
            self.cat_feature_indices_ = []
        else:
            cat_cols = [
                c for c in X_trans.columns
                if X_trans[c].dtype == "object" or str(X_trans[c].dtype).startswith("category")
            ]
            self.cat_feature_indices_ = [X_trans.columns.get_loc(c) for c in cat_cols]

            # XGBoost categorical support requires pandas 'category' dtype
            if cat_cols:
                X_trans[cat_cols] = X_trans[cat_cols].astype("category")

        self.cat_cols_ = cat_cols

        if self.xgb_verbose:
            print("Categorical columns:", self.cat_cols_)
            print("Categorical feature indices:", self.cat_feature_indices_)

        # 5) build XGBoost params
        xgb_params = dict(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            gamma=self.gamma,
            tree_method=self.tree_method,
            max_bin=self.max_bin,
            objective=self.objective,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            verbosity=2 if self.xgb_verbose else 0,
            # Avoid nested parallelism: outer CV controls parallelism
            n_jobs=1,
            enable_categorical=True,
        )

        # 6) train XGBRegressor
        self.model_ = XGBRegressor(**xgb_params)
        self.model_.fit(X_trans, y)

        # Optional: quick training metric
        if self.xgb_verbose:
            train_pred = self.model_.predict(X_trans)
            train_mae = mean_absolute_error(y, train_pred)
            print(f"Training MAE: {train_mae:.4f}")

        return self

    def predict(self, X):
        """Apply the same feature engineering and XGBoost model to new data."""
        if self.fe_ is None or self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_trans = self.fe_.transform(X)

        # Re-apply dtype casting for categoricals
        if self.cat_cols_:
            X_trans[self.cat_cols_] = X_trans[self.cat_cols_].astype("category")

        return self.model_.predict(X_trans)