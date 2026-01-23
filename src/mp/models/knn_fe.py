import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error

from mp.feature_engineer import FeatureEngineer


class KNNFEModel(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        # ---- KNN params (tunable) ----
        n_neighbors: int = 8,
        weights: str = "distance",
        p: int = 1,
        algorithm: str = "auto",
        leaf_size: int = 63,
        n_jobs: int = 1,           
        knn_verbose: bool = False,

        # ---- FeatureEngineer hyperparams (tunable) ----
        corr_threshold: float = 0.945,
        min_fp_freq: float = 0.0261,
        max_fp_freq: float = 0.869,
        fe_verbose: bool = False,
        target_col: str = "mpC",

        # ---- Optional FE behavior ----
        normalize: bool = True,       
        onehot_cats: bool = True,    
        cat_cols: list[str] | None = None,

        # ---- Column conventions ----
        smiles_col: str = "smiles",
        fp_prefix: str = "FP",

        # ---- Consistency ----
        random_state: int = 42,
    ):
        # KNN
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.knn_verbose = knn_verbose

        # FE thresholds + behavior
        self.corr_threshold = corr_threshold
        self.min_fp_freq = min_fp_freq
        self.max_fp_freq = max_fp_freq
        self.fe_verbose = fe_verbose
        self.target_col = target_col

        self.normalize = normalize
        self.onehot_cats = onehot_cats
        self.cat_cols = cat_cols if cat_cols is not None else ["flag", "source"]

        self.smiles_col = smiles_col
        self.fp_prefix = fp_prefix

        self.random_state = random_state

        # learned in fit()
        self.fe_ = None
        self.model_ = None
        self.feature_names_in_ = None
        self.feature_names_ = None
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y):
        """Fit FeatureEngineer on training data, then fit KNN on engineered matrix."""
        if y is None:
            raise ValueError("y cannot be None for fit().")

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

        if not isinstance(X_trans, pd.DataFrame):
            X_trans = pd.DataFrame(X_trans)

        # KNN requires purely numeric input
        X_trans = X_trans.apply(pd.to_numeric, errors="coerce").fillna(0.0)

     
       # Store training column order for deterministic alignment in predict()
        self.feature_names_ = X_trans.columns.tolist()

        # Train KNN
        self.model_ = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            p=self.p,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X_trans, y)

        self.fitted_ = True

        if self.knn_verbose:
            train_pred = self.model_.predict(X_trans)
            train_mae = mean_absolute_error(y, train_pred)
            print("KNN: transformed shape:", X_trans.shape)
            print(f"KNN: training MAE = {train_mae:.4f}")

        return self

    def predict(self, X: pd.DataFrame):
        """Apply FeatureEngineer then predict with trained KNN (schema-aligned to training)."""
        if not self.fitted_ or self.fe_ is None or self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_trans = self.fe_.transform(X)
        if not isinstance(X_trans, pd.DataFrame):
            X_trans = pd.DataFrame(X_trans)

        X_trans = X_trans.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")

        # Align to training columns: drop extras, add missing as 0
        if self.feature_names_ is not None:
            X_trans = X_trans.reindex(columns=self.feature_names_, fill_value=0.0)

        return self.model_.predict(X_trans)