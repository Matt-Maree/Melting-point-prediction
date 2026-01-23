from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(data_set: str, root: Path) -> pd.DataFrame:
    """Load a single melting-point dataset (full or curated) and apply basic cleaning.

    - Reads the raw CSV from data/raw/
    - Keeps only columns needed downstream (SMILES, target, source, dataset flag)
    - Drops rows missing SMILES or melting point
    - Normalises missing source labels
    """
    root = Path(root)

    if data_set == "Full":
        # Larger, noisier dataset; drop entries flagged as "do not use"
        path = root / "data" / "raw" / "BradleyMeltingPointDataset.csv"
        data = pd.read_csv(path).set_index("key")
        data = data[data["donotuse"].isna()]
        data["flag"] = "full"

    elif data_set == "Curated":
        # Smaller, higher-quality subset
        path = root / "data" / "raw" / "BradleyDoublePlusGoodMeltingPointDataset.csv"
        data = pd.read_csv(path).set_index("key")
        data["flag"] = "curated"

    else:
        raise ValueError("data_set must be either 'Full' or 'Curated'")

    # Keep only the fields used by featurization / modelling
    data = data[["smiles", "mpC", "source", "flag"]]

    # Drop rows with missing SMILES or target melting point
    data = data.dropna(subset=["smiles", "mpC"])

    # Fill missing source information with a single category
    data["source"] = data["source"].fillna("unknown")

    return data


def get_data(root: Path, rare_source_threshold: int = 50) -> pd.DataFrame:
    """Load both datasets, merge, deduplicate by SMILES, and reduce source cardinality.

    Deduplication rule:
    - If the same SMILES appears in both datasets, keep the curated entry.

    Source grouping:
    - Collapse rare source labels into "other" to keep categorical handling manageable.
    """
    # Load both raw datasets with consistent cleaning
    full = load_data("Full", root=root)
    curated = load_data("Curated", root=root)
    merged = pd.concat([full, curated], ignore_index=True)

    # Ensure curated rows win when SMILES duplicates exist
    merged["priority"] = merged["flag"].map({"curated": 0, "full": 1})
    merged = merged.sort_values(["smiles", "priority"], ascending=True)
    merged = merged.drop_duplicates(subset="smiles", keep="first")
    merged = merged.drop(columns=["priority"])

    # Reduce the number of unique 'source' categories by grouping rare values
    value_counts = merged["source"].value_counts()
    rare_sources = value_counts[value_counts < rare_source_threshold].index
    merged.loc[merged["source"].isin(rare_sources), "source"] = "other"

    return merged