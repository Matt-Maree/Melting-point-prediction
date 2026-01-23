import numpy as np
import pandas as pd
import random

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split

from mp.feature_engineer import FeatureEngineer

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress INFO + WARNING

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class NNFEModel(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        # ---- FeatureEngineer hyperparams (tunable) ----
        corr_threshold: float = 0.96,
        min_fp_freq: float = 0.006,
        max_fp_freq: float = 0.72,
        fe_verbose: bool = False,
        target_col: str = "mpC",
        smiles_col: str = "smiles",
        fp_prefix: str = "FP",

        # ---- NN training hyperparams ----
        learning_rate: float = 3.5e-4,
        batch_size: int = 128,
        epochs: int = 500,
        patience: int = 40,
        lr_patience: int = 10,
        min_lr: float = 1e-5,
        validation_split: float = 0.2,  # NOTE: applied via explicit split when using tf.data

        # ---- Regularisation ----
        l2_strength: float = 3e-5,
        dropout: float = 0.30,

        # ---- Reproducibility ----
        random_state: int = 42,
    ):
        # FE params
        self.corr_threshold = corr_threshold
        self.min_fp_freq = min_fp_freq
        self.max_fp_freq = max_fp_freq
        self.fe_verbose = fe_verbose
        self.target_col = target_col
        self.smiles_col = smiles_col
        self.fp_prefix = fp_prefix

        # NN params
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr_patience = lr_patience
        self.min_lr = min_lr
        self.validation_split = validation_split

        # regularisation
        self.l2_strength = l2_strength
        self.dropout = dropout

        # reproducibility
        self.random_state = random_state

        # learned in fit()
        self.fe_ = None
        self.model_ = None
        self.feature_names_in_ = None
        self.feature_names_ = None
        self.history_ = None

    def _set_seeds(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

    def _build_model(self, input_dim: int) -> tf.keras.Model:
        l2_reg = regularizers.l2(self.l2_strength)

        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(256, activation="relu", kernel_regularizer=l2_reg),
            Dropout(self.dropout),
            Dense(256, activation="relu", kernel_regularizer=l2_reg),
            Dropout(self.dropout),
            Dense(128, activation="relu", kernel_regularizer=l2_reg),
            Dense(1),
        ])

        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(optimizer=opt, loss="mae", metrics=["mae"])
        return model

    def fit(self, X: pd.DataFrame, y):
        """Fit FeatureEngineer on training data, then train a Keras regressor."""
        self._set_seeds()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        self.feature_names_in_ = list(X.columns)

        # Feature engineering: NN benefits from scaling + OHE
        self.fe_ = FeatureEngineer(
            corr_threshold=self.corr_threshold,
            min_fp_freq=self.min_fp_freq,
            max_fp_freq=self.max_fp_freq,
            verbose=self.fe_verbose,
            normalize=True,
            onehot_cats=True,
            cat_cols=["flag", "source"],
            target_col=self.target_col,
            smiles_col=self.smiles_col,
            fp_prefix=self.fp_prefix,
        )

        # Attach target so FE can safely exclude it from selection logic
        df_train = X.assign(**{self.target_col: y.to_numpy()})
        self.fe_.fit(df_train)

        X_trans = self.fe_.transform(X)
        if not isinstance(X_trans, pd.DataFrame):
            X_trans = pd.DataFrame(X_trans)

        self.feature_names_ = list(X_trans.columns)

        # TensorFlow-friendly arrays
        X_np = X_trans.to_numpy(dtype=np.float32, copy=False)
        y_np = y.to_numpy(dtype=np.float32, copy=False).reshape(-1)

        self.model_ = self._build_model(input_dim=X_np.shape[1])

        early_stop = EarlyStopping(
            monitor="val_mae",
            patience=self.patience,
            restore_best_weights=True,
            verbose=0,
        )
        lr_reduce = ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.8,
            patience=self.lr_patience,
            min_lr=self.min_lr,
            verbose=0,
        )

        # Create an explicit train/val split, then pass validation_data to model.fit().
        if not (0.0 < self.validation_split < 1.0):
            raise ValueError("validation_split must be between 0 and 1 (exclusive).")

        X_tr, X_va, y_tr, y_va = train_test_split(
            X_np,
            y_np,
            test_size=self.validation_split,
            random_state=self.random_state,
        )

        train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
        train_ds = train_ds.shuffle(
            buffer_size=min(len(X_tr), 10_000),
            seed=self.random_state,
            reshuffle_each_iteration=True,
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((X_va, y_va))
        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        self.history_ = self.model_.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=[early_stop, lr_reduce],
            verbose=0,
        )

        return self

    def predict(self, X: pd.DataFrame):
        """Apply fitted FE then predict with trained NN (schema-aligned to training)."""
        if self.fe_ is None or self.model_ is None or self.feature_names_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_trans = self.fe_.transform(X)
        if not isinstance(X_trans, pd.DataFrame):
            X_trans = pd.DataFrame(X_trans)

        X_trans = X_trans.reindex(columns=self.feature_names_, fill_value=0.0)

        X_np = X_trans.to_numpy(dtype=np.float32, copy=False)
        return self.model_.predict(X_np, verbose=0).reshape(-1)