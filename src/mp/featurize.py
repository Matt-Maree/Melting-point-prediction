import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors, rdmolops

# Silence noisy RDKit warnings in notebooks / logs
RDLogger.DisableLog("rdApp.*")

# Morgan fingerprint settings (ECFP-style)
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_NBITS)

# Allowed elements for "neutral organic" filtering
ALLOWED_ELEMENTS = {
    "C", "H", "N", "O", "S", "P",
    "F", "Cl", "Br", "I",
    "B", "Si",
}


def safe_mol_from_smiles(s: str):
    """Safely parse a SMILES string into an RDKit Mol.

    Returns:
        RDKit Mol on success, or None if SMILES is missing/invalid.
    """
    if pd.isna(s):
        return None
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


def is_neutral_organic(mol) -> bool:
    if mol is None:
        return False

    # Exclude charged molecules
    if rdmolops.GetFormalCharge(mol) != 0:
        return False

    # Element whitelist + must contain carbon
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    if "C" not in elements or not elements.issubset(ALLOWED_ELEMENTS):
        return False

    # Require at least one carbon with at least one (implicit or explicit) hydrogen
    return any(
        a.GetAtomicNum() == 6 and a.GetTotalNumHs(includeNeighbors=True) > 0
        for a in mol.GetAtoms()
    )


def get_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generate RDKit descriptors and Morgan fingerprints from a SMILES dataframe.

    Expected input columns:
        - smiles: SMILES string
        - mpC: target melting point (kept/used downstream but not required for featurization)
        - source / flag: provenance columns (passed through until later processing)

    Output:
        - RDKit descriptor columns (continuous)
        - Morgan fingerprint bit columns FP_0..FP_2047 (binary)
        - Drops intermediate 'mol' and 'smiles' columns before returning
    """
    data = data.copy()

    # Convert SMILES to RDKit Mol objects
    data["mol"] = data["smiles"].apply(safe_mol_from_smiles)

    # Drop invalid SMILES that failed to parse
    valid_mask = data["mol"].notnull()
    print(f"Dropping {(~valid_mask).sum()} rows with invalid SMILES.")
    data = data[valid_mask].copy()

    # Drop charged and non-organic molecules (as defined above)
    data["is_neutral_organic"] = data["mol"].apply(is_neutral_organic)
    n_dropped = (~data["is_neutral_organic"]).sum()
    print(f"Dropping {n_dropped} charged and non-organic compounds")
    data = data[data["is_neutral_organic"]].drop(columns=["is_neutral_organic"]).copy()

    # Simple atom count helper functions (adds some basic composition info)
    count_N = lambda m: sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 7)
    count_O = lambda m: sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 8)
    count_S = lambda m: sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 16)
    count_hal = lambda m: sum(1 for a in m.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53))

    # Descriptor functions to compute (2D physchem + ring/shape/connectivity)
    descs = {
        "MolWt": Descriptors.MolWt,
        "LogP": Descriptors.MolLogP,
        "TPSA": Descriptors.TPSA,
        "HBD": Descriptors.NumHDonors,
        "HBA": Descriptors.NumHAcceptors,
        "RotB": Descriptors.NumRotatableBonds,
        "RingCount": Descriptors.RingCount,
        "FracCSP3": rdMolDescriptors.CalcFractionCSP3,
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
        "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
        "BertzCT": Descriptors.BertzCT,
        "Kappa1": Descriptors.Kappa1,
        "Kappa2": Descriptors.Kappa2,
        "Kappa3": Descriptors.Kappa3,
        "Chi1v": Descriptors.Chi1v,
        "LabuteASA": rdMolDescriptors.CalcLabuteASA,
        "HeavyAtomCount": Descriptors.HeavyAtomCount,
        "NumAromaticAtoms": lambda m: sum(a.GetIsAromatic() for a in m.GetAtoms()),
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
        "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings,
        "NumAromaticHeterocycles": rdMolDescriptors.CalcNumAromaticHeterocycles,
        "NumAromaticCarbocycles": rdMolDescriptors.CalcNumAromaticCarbocycles,
        "MolMR": Descriptors.MolMR,
        "NumN": count_N,
        "NumO": count_O,
        "NumS": count_S,
        "NumHalogen": count_hal,

        # Connectivity indices (branching / shape)
        "Chi0v": Descriptors.Chi0v,
        "Chi0n": Descriptors.Chi0n,
        "Chi1n": Descriptors.Chi1n,
        "Chi2v": Descriptors.Chi2v,
        "Chi2n": Descriptors.Chi2n,

        # Surface area fragments (sometimes weakly predictive)
        "SlogP_VSA1": Descriptors.SlogP_VSA1,
        "SlogP_VSA2": Descriptors.SlogP_VSA2,

        # Partial charge summaries (can be slower; keep if stable in your env)
        "MaxPartialCharge": Descriptors.MaxPartialCharge,
        "MinPartialCharge": Descriptors.MinPartialCharge,

        # Shape / alpha descriptor
        "HallKierAlpha": Descriptors.HallKierAlpha,
    }

    # Compute RDKit descriptors column-by-column
    for name, fn in descs.items():
        data[name] = data["mol"].apply(fn)

    # Generate Morgan fingerprints and expand bits into FP_0..FP_2047 columns
    def get_morgan_fp(m):
        fp = morgan_gen.GetFingerprint(m)
        return fp.ToList()

    fp_df = data["mol"].apply(get_morgan_fp).apply(pd.Series)
    fp_df.columns = [f"FP_{i}" for i in range(fp_df.shape[1])]
    data = pd.concat([data, fp_df], axis=1)

    # Drop intermediate columns (mol objects and original SMILES)
    data = data.drop(columns=["mol", "smiles"]).reset_index(drop=True)

    print(f"{len(data)} entries remaining, with {data.shape[1]} columns")
    return data