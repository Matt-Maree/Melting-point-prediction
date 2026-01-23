import pandas as pd

from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error

from mp.feature_engineer import FeatureEngineer


class CatBoostFEModel(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        # CatBoost params (tunable)
        depth: int = 11,
        learning_rate: float = 0.0337,
        n_estimators: int = 2032,
        l2_leaf_reg: float = 19.3,
        bagging_temperature: float = 0.432,
        random_strength: float = 0.491,
        grow_policy: str = "Depthwise",
        rsm: float = 0.815,
        min_data_in_leaf: int = 11,
        leaf_estimation_iterations: int = 2,
        loss_function: str = "MAE",
        random_state: int = 42,
        cb_verbose: bool = False,

        # FeatureEngineer hyperparams (tunable)
        corr_threshold: float = 0.879,
        min_fp_freq: float = 0.0133,
        max_fp_freq: float = 0.936,
        fe_verbose: bool = False,
        target_col: str = "mpC",

        # Optional FE behaviour
        normalize: bool = False,
        onehot_cats: bool = False,
        cat_cols: list | None = None,
        smiles_col: str = "smiles",
        fp_prefix: str = "FP",
    ):
        # CatBoost params
        self.depth = depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.l2_leaf_reg = l2_leaf_reg
        self.bagging_temperature = bagging_temperature
        self.random_strength = random_strength
        self.grow_policy = grow_policy
        self.rsm = rsm
        self.min_data_in_leaf = min_data_in_leaf
        self.leaf_estimation_iterations = leaf_estimation_iterations
        self.loss_function = loss_function
        self.random_state = random_state
        self.cb_verbose = cb_verbose

        # FE params
        self.corr_threshold = corr_threshold
        self.min_fp_freq = min_fp_freq
        self.max_fp_freq = max_fp_freq
        self.fe_verbose = fe_verbose
        self.target_col = target_col

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
        """Fit FeatureEngineer + CatBoostRegressor."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Ensure alignment and 1D target
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        self.feature_names_in_ = list(X.columns)

        # Attach target so FeatureEngineer can safely exclude it from feature-selection logic
        df_with_target = X.copy()
        df_with_target[self.target_col] = y

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
        self.fe_.fit(df_with_target)
        X_trans = self.fe_.transform(X)

        # Categorical handling:
        if self.onehot_cats:
            cat_cols = []
            self.cat_feature_indices_ = []
        else:
            cat_cols = [
                c for c in X_trans.columns
                if X_trans[c].dtype == "object" or str(X_trans[c].dtype).startswith("category")
            ]
            self.cat_feature_indices_ = [X_trans.columns.get_loc(c) for c in cat_cols]

        self.cat_cols_ = cat_cols

        if self.cb_verbose:
            print("Categorical columns:", self.cat_cols_)
            print("Categorical feature indices:", self.cat_feature_indices_)

        # Train CatBoost on engineered feature matrix
        cb_params = dict(
            random_state=self.random_state,
            loss_function=self.loss_function,
            depth=self.depth,
            learning_rate=self.learning_rate,
            min_data_in_leaf=self.min_data_in_leaf,
            leaf_estimation_iterations=self.leaf_estimation_iterations,
            n_estimators=self.n_estimators,
            l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength,
            bagging_temperature=self.bagging_temperature,
            grow_policy=self.grow_policy,
            rsm=self.rsm,
            verbose=self.cb_verbose,
        )

        self.model_ = CatBoostRegressor(**cb_params)
        self.model_.fit(X_trans, y, cat_features=self.cat_feature_indices_, verbose=self.cb_verbose)

        if self.cb_verbose:
            train_pred = self.model_.predict(X_trans)
            train_mae = mean_absolute_error(y, train_pred)
            print(f"Training MAE: {train_mae:.4f}")

        return self

    def predict(self, X):
        """Apply the fitted FeatureEngineer and CatBoost model to new data."""
        if self.fe_ is None or self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_trans = self.fe_.transform(X)
        return self.model_.predict(X_trans)