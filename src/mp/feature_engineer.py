from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Feature engineering helper for chemical datasets that include:
      - RDKit descriptor columns (numeric)
      - optional Morgan fingerprint bit columns named like "FP0", "FP1", ...
      - optional SMILES column

    What it does:
      1) Adds engineered ratio/fraction features based on common RDKit descriptors.
      2) Learns which FP columns to drop based on frequency thresholds.
      3) Learns which numeric (non-FP) features to drop based on correlation threshold.
         The "which one to drop?" rule is:
            (a) lower FEATURE_PRIORITY drops first
            (b) if tie: feature appearing in MORE correlated pairs drops first
            (c) if tie: deterministic by column order (later column drops)
      4) Learns training-set numeric NaN fill values (means).
      5) Optionally learns normalisation stats (mean/std) for numeric features.
      6) Optionally learns one-hot encoding structure for categorical columns.
      7) Ensures transform() outputs a consistent column set + order.

    Notes:
      - Correlation logic is UNSUPERVISED: uses feature-feature correlations only (never the target).
      - fit() called on TRAINING data only, then transform() on train/val/test.
    """

    FEATURE_PRIORITY: Dict[str, int] = {
        # Core physicochemical
        "MolWt": 5,
        "LogP": 5,
        "TPSA": 5,
        "HBD": 5,
        "HBA": 5,
        # Structural / flexibility
        "RingCount": 4,
        "NumAromaticRings": 4,
        "HeavyAtomCount": 4,
        "RotB": 4,
        "FracCSP3": 4,
        "NumSaturatedRings": 4,
        "NumAliphaticRings": 4,
        "NumAromaticHeterocycles": 4,
        "NumAromaticCarbocycles": 4,
        # Element counts
        "NumN": 3,
        "NumO": 3,
        "NumS": 3,
        "NumHalogen": 3,
        "NumHeteroatoms": 3,
        # Derived features (added by _add_engineered_features)
        "HBD_HBA_sum": 2,
        "HBD_HBA_ratio": 2,
        "HBD_per_heavy": 2,
        "HBA_per_heavy": 2,
        "TPSA_per_MW": 2,
        "TPSA_per_heavy": 2,
        "HetFrac": 2,
        "AromaticFrac": 2,
        "AromaticRingFrac": 2,
        "RingDensity": 2,
        "MW_per_ASA": 2,
        "Heavy_per_ASA": 2,
        # Graph indices
        "Chi0v": 2,
        "Chi0n": 2,
        "Chi1n": 2,
        "Chi2v": 2,
        "Chi2n": 2,
        "Kappa1": 2,
        "Kappa2": 2,
        "Kappa3": 2,
        "LabuteASA": 2,
        "MolMR": 2,
        # Others
        "SlogP_VSA1": 1,
        "SlogP_VSA2": 1,
        "BertzCT": 1,
        "Chi1v": 1,
        "HallKierAlpha": 1,
        "MinPartialCharge": 1,
        "MaxPartialCharge": 1,
    }

    def __init__(
        self,
        corr_threshold: float = 1.00,
        min_fp_freq: float = 0.00,
        max_fp_freq: float = 1.00,
        verbose: bool = True,
        normalize: bool = False,
        onehot_cats: bool = False,
        cat_cols: Optional[List[str]] = None,
        target_col: str = "mpC",
        smiles_col: str = "smiles",
        fp_prefix: str = "FP",
        feature_priority: Optional[Dict[str, int]] = None,
    ):
        # thresholds / flags
        self.corr_threshold = float(corr_threshold)
        self.min_fp_freq = float(min_fp_freq)
        self.max_fp_freq = float(max_fp_freq)
        self.verbose = bool(verbose)
        self.normalize = bool(normalize)
        self.onehot_cats = bool(onehot_cats)
        self.cat_cols = cat_cols

        # naming conventions
        self.target_col = target_col
        self.smiles_col = smiles_col
        self.fp_prefix = fp_prefix

        # allow overriding priorities
        self.feature_priority = feature_priority if feature_priority is not None else self.FEATURE_PRIORITY

        # learned attributes populated by fit()
        self.fp_drop_cols_: List[str] = []
        self.corr_drop_cols_: List[str] = []
        self.constant_drop_cols_: List[str] = []  # now populated (see get_corr_drop_cols)
        self.fitted_: bool = False

        # for normalisation
        self.norm_cols_: Optional[List[str]] = None
        self.norm_means_: Optional[pd.Series] = None
        self.norm_stds_: Optional[pd.Series] = None

        # for one-hot encoding
        self.cat_cols_: List[str] = []
        self.ohe_dummy_cols_: List[str] = []
        self.feature_order_: Optional[List[str]] = None

        # for NaN filling (numeric cols)
        self.fill_values_: Optional[pd.Series] = None

    # ---------- 0. Add engineered descriptor features ---------- #
    @staticmethod
    def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional ratio/fraction features derived from RDKit descriptors.
        """
        df = df.copy()
        eps = 1e-6  # avoid div-by-zero

        # Helpers
        def has(cols: set[str]) -> bool:
            return cols.issubset(df.columns)

        # H-bonding / polarity related
        if has({"HBD", "HBA"}):
            df["HBD_HBA_sum"] = df["HBD"] + df["HBA"]
            df["HBD_HBA_ratio"] = df["HBD"] / (df["HBA"] + eps)

        if has({"HBD", "HeavyAtomCount"}):
            df["HBD_per_heavy"] = df["HBD"] / (df["HeavyAtomCount"] + eps)

        if has({"HBA", "HeavyAtomCount"}):
            df["HBA_per_heavy"] = df["HBA"] / (df["HeavyAtomCount"] + eps)

        if has({"TPSA", "MolWt"}):
            df["TPSA_per_MW"] = df["TPSA"] / (df["MolWt"] + eps)

        if has({"TPSA", "HeavyAtomCount"}):
            df["TPSA_per_heavy"] = df["TPSA"] / (df["HeavyAtomCount"] + eps)

        # Heteroatom & aromaticity fractions
        if has({"NumHeteroatoms", "HeavyAtomCount"}):
            df["HetFrac"] = df["NumHeteroatoms"] / (df["HeavyAtomCount"] + eps)

        if has({"NumAromaticAtoms", "HeavyAtomCount"}):
            df["AromaticFrac"] = df["NumAromaticAtoms"] / (df["HeavyAtomCount"] + eps)

        if has({"NumAromaticRings", "RingCount"}):
            df["AromaticRingFrac"] = df["NumAromaticRings"] / (df["RingCount"] + eps)

        if has({"RingCount", "HeavyAtomCount"}):
            df["RingDensity"] = df["RingCount"] / (df["HeavyAtomCount"] + eps)

        # Packing / density-ish proxies
        if has({"MolWt", "LabuteASA"}):
            df["MW_per_ASA"] = df["MolWt"] / (df["LabuteASA"] + eps)

        if has({"HeavyAtomCount", "LabuteASA"}):
            df["Heavy_per_ASA"] = df["HeavyAtomCount"] / (df["LabuteASA"] + eps)

        return df

    # ---------- helpers ---------- #
    def _is_fp_col(self, col: str) -> bool:
        return isinstance(col, str) and col.startswith(self.fp_prefix)

    def _prio(self, col: str) -> int:
        """Feature priority for correlation-dropping decisions (unknown -> 0)."""
        return int(self.feature_priority.get(col, 0))

    # ---------- 1. Compute columns to drop (fit-time logic) ---------- #
    def get_fp_drop_cols(self, df: pd.DataFrame) -> List[str]:
        """Drop FP columns whose mean frequency is < min_fp_freq or > max_fp_freq."""
        fp_cols = [c for c in df.columns if self._is_fp_col(c)]
        if not fp_cols:
            if self.verbose:
                print("No FP columns found for fingerprint frequency filter.")
            return []

        fp_mat = df[fp_cols].apply(pd.to_numeric, errors="coerce")
        freq_fp = fp_mat.mean(axis=0)

        drop_fp_cols = freq_fp[(freq_fp < self.min_fp_freq) | (freq_fp > self.max_fp_freq)].index.tolist()

        if self.verbose:
            print("Fingerprint frequency filter:")
            print(f" - thresholds: min={self.min_fp_freq}, max={self.max_fp_freq}")
            print(f" - dropping {len(drop_fp_cols)} FP columns")
            print(f" - remaining FP columns: {len(fp_cols) - len(drop_fp_cols)}/{len(fp_cols)}")

        return drop_fp_cols

    def get_corr_drop_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Return NON-FP numeric feature columns to drop based on absolute correlation threshold.
        """
        work = df.copy()

        # Remove target/smiles if present (never used for correlation logic)
        for col in (self.target_col, self.smiles_col):
            if col in work.columns:
                work = work.drop(columns=[col])

        # Remove fingerprint columns from correlation analysis
        fp_cols = [c for c in work.columns if self._is_fp_col(c)]
        if fp_cols:
            work = work.drop(columns=fp_cols)

        # Numeric only
        X_num = work.select_dtypes(include=[np.number]).copy()
        cols = list(X_num.columns)

        if not cols:
            if self.verbose:
                print("Correlation filter: no numeric (non-FP) features found.")
            self.constant_drop_cols_ = []
            return []

        # Drop constant numeric columns (including all-NaN -> nunique=0)
        nunique = X_num.nunique(dropna=False)
        constant_cols = nunique[nunique <= 1].index.tolist()
        self.constant_drop_cols_ = constant_cols  # <--- minor change: persist separately

        if constant_cols:
            X_num = X_num.drop(columns=constant_cols)
            cols = list(X_num.columns)

        if len(cols) < 2:
            if self.verbose and constant_cols:
                print(f"Correlation filter: dropped {len(constant_cols)} constant numeric columns.")
            return sorted(constant_cols)

        # Compute absolute correlation matrix
        corr = X_num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        pairs: List[tuple[str, str, float]] = []
        for j in upper.columns:
            s = upper[j]
            hits = s[s > self.corr_threshold]
            for i, v in hits.items():
                pairs.append((i, j, float(v)))

        if not pairs:
            if self.verbose:
                print(
                    f"Correlation filter: no pairs above threshold {self.corr_threshold}. "
                    f"Constant cols dropped: {len(constant_cols)}"
                )
            return sorted(constant_cols)

        # Redundancy degree
        counts = (
            pd.Series([p[0] for p in pairs] + [p[1] for p in pairs])
            .value_counts()
            .to_dict()
        )

        # Deterministic tie-break order
        order = {c: k for k, c in enumerate(cols)}

        def choose_drop(a: str, b: str) -> str:
            pa, pb = self._prio(a), self._prio(b)
            if pa != pb:
                return a if pa < pb else b
            ca, cb = counts.get(a, 0), counts.get(b, 0)
            if ca != cb:
                return a if ca > cb else b
            return a if order[a] > order[b] else b

        to_drop = set(constant_cols)
        for a, b, _ in pairs:
            to_drop.add(choose_drop(a, b))

        # Stable order: use original numeric column order from work (before dropping fp/target/smiles)
        numeric_order = list(work.select_dtypes(include=[np.number]).columns)
        drop_list = [c for c in numeric_order if c in to_drop]
        for c in sorted(constant_cols):
            if c not in drop_list:
                drop_list.append(c)

        if self.verbose:
            total_numeric = len(numeric_order)
            remaining_numeric = total_numeric - len(set(drop_list))
            print("Correlation filter:")
            print(f" - threshold: {self.corr_threshold}")
            print(f" - correlated pairs found: {len(pairs)}")
            print(f" - dropping {len(set(drop_list))} numeric cols (includes {len(constant_cols)} constants)")
            print(f" - remaining numeric cols: {remaining_numeric}/{total_numeric}")

        return drop_list

    # ---------- 2. Apply drops ---------- #
    @staticmethod
    def drop_columns(df: pd.DataFrame, cols_to_drop: Sequence[str]) -> pd.DataFrame:
        """Drop columns safely (ignores missing columns)."""
        return df.drop(columns=list(cols_to_drop), errors="ignore")

    # ---------- 3. Fit / transform ---------- #
    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        Learn fit-time artifacts from TRAINING data only:
          - FP frequency drop list
          - correlated/constant numeric drop list
          - numeric NaN fill means
          - normalisation stats (optional)
          - one-hot structure + final column order (optional)
        """
        # Add engineered features (robust to missing inputs)
        df_eng = self._add_engineered_features(df)

        # Learn which columns to drop using training data only (prevents leakage of distribution stats)
        fp_drop = self.get_fp_drop_cols(df_eng)
        df_tmp = self.drop_columns(df_eng, fp_drop)

        corr_drop = self.get_corr_drop_cols(df_tmp)

        self.fp_drop_cols_ = fp_drop
        self.corr_drop_cols_ = corr_drop
        self.fitted_ = True

        # Apply all drops to get the "base" feature set
        all_drops = set(fp_drop) | set(corr_drop)

        if self.smiles_col in df_tmp.columns:
            all_drops.add(self.smiles_col)
        if self.target_col in df_tmp.columns:
            all_drops.add(self.target_col)

        df_after_drop = self.drop_columns(df_tmp, all_drops)

        if self.verbose:
            print(f"Total cols after engineered features: {df_eng.shape[1]}")
            print(f"Total cols after drops: {df_after_drop.shape[1]}")

        # Store training-set means for numeric NaN imputation (applied consistently at transform time)
        self.fill_values_ = df_after_drop.mean(numeric_only=True)

        if self.verbose:
            na_cols = df_after_drop.columns[df_after_drop.isna().any()].tolist()
            if na_cols:
                print(f"{len(na_cols)} feature cols contain NaNs (numeric NaNs filled with training means).")

        # Fill numeric NaNs in a working copy before computing norm / OHE structures
        df_for_stats = df_after_drop.copy()
        if self.fill_values_ is not None and not self.fill_values_.empty:
            intersect = [c for c in df_for_stats.columns if c in self.fill_values_.index]
            if intersect:
                df_for_stats[intersect] = df_for_stats[intersect].fillna(self.fill_values_[intersect])

        # Learn normalisation stats (optional)
        if self.normalize:
            num_cols = [
                c for c in df_for_stats.select_dtypes(include=[np.number]).columns
                if not self._is_fp_col(c)
            ]
            self.norm_cols_ = num_cols

            if num_cols:
                self.norm_means_ = df_for_stats[num_cols].mean()
                self.norm_stds_ = df_for_stats[num_cols].std().replace(0.0, 1.0)

                if self.verbose:
                    print(f"Normalisation enabled for {len(num_cols)} numeric feature columns.")
            else:
                self.norm_means_ = None
                self.norm_stds_ = None
                if self.verbose:
                    print("Normalisation enabled but no numeric feature columns found.")

        # Learn one-hot structure + final feature order (optional)
        if self.onehot_cats:
            if self.cat_cols is not None:
                cat_cols = [c for c in self.cat_cols if c in df_after_drop.columns]
            else:
                cat_cols = [
                    c for c in df_after_drop.columns
                    if (df_after_drop[c].dtype == "object" or str(df_after_drop[c].dtype).startswith("category"))
                ]
            self.cat_cols_ = cat_cols

            if cat_cols:
                df_ohe = pd.get_dummies(df_after_drop, columns=cat_cols, drop_first=False)

                original_cols = set(df_after_drop.columns)
                self.ohe_dummy_cols_ = [c for c in df_ohe.columns if c not in original_cols]

                # Record final feature order so downstream models see a stable schema
                self.feature_order_ = list(df_ohe.columns)

                if self.verbose:
                    print(f"One-hot encoding enabled for {len(cat_cols)} categorical columns.")
                    print(f"Created {len(self.ohe_dummy_cols_)} dummy columns.")
            else:
                self.ohe_dummy_cols_ = []
                self.feature_order_ = list(df_after_drop.columns)
                if self.verbose:
                    print("One-hot encoding enabled but no categorical columns found.")
        else:
            self.cat_cols_ = []
            self.ohe_dummy_cols_ = []
            self.feature_order_ = list(df_after_drop.columns)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned feature engineering to any dataset.

        Guarantees:
          - same column set and order as training output
          - target column excluded
        """
        if not self.fitted_:
            raise RuntimeError("FeatureEngineer must be fitted before calling transform().")

        df_eng = self._add_engineered_features(df)

        # Drop learned columns + SMILES + target (if present)
        all_drops = set(self.fp_drop_cols_) | set(self.corr_drop_cols_)
        if self.smiles_col in df_eng.columns:
            all_drops.add(self.smiles_col)
        if self.target_col in df_eng.columns:
            all_drops.add(self.target_col)

        df_clean = self.drop_columns(df_eng, all_drops)

        # Fill numeric NaNs using training means
        if self.fill_values_ is not None and not self.fill_values_.empty:
            intersect = [c for c in df_clean.columns if c in self.fill_values_.index]
            if intersect:
                df_clean[intersect] = df_clean[intersect].fillna(self.fill_values_[intersect])

        # Apply normalisation
        if (
            self.normalize
            and self.norm_cols_ is not None
            and self.norm_means_ is not None
            and self.norm_stds_ is not None
        ):
            for c in self.norm_cols_:
                if c in df_clean.columns:
                    df_clean[c] = (df_clean[c] - self.norm_means_[c]) / self.norm_stds_[c]

        # One-hot encoding (optional)
        if self.onehot_cats and self.cat_cols_:
            df_ohe = pd.get_dummies(df_clean, columns=self.cat_cols_, drop_first=False)

            # Ensure all dummy columns from training exist
            for col in self.ohe_dummy_cols_:
                if col not in df_ohe.columns:
                    df_ohe[col] = 0

            # Enforce training-time feature schema (drops unseen dummy columns)
            df_ohe = df_ohe.reindex(columns=self.feature_order_, fill_value=0)
            return df_ohe

        # No OHE: enforce training-time feature schema
        if self.feature_order_ is not None:
            df_clean = df_clean.reindex(columns=self.feature_order_, fill_value=0)

        return df_clean

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method: fit on df, then transform df."""
        return self.fit(df).transform(df)

    def get_feature_names_out(self) -> np.ndarray:
        """Sklearn-style helper: feature names after transform()."""
        if not self.fitted_ or self.feature_order_ is None:
            raise RuntimeError("Call fit() before get_feature_names_out().")
        return np.array(self.feature_order_, dtype=object)