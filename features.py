"""Feature transforms applied after data loading.

This file is edited by the agent. The harness calls transform().
"""

import pandas as pd
import numpy as np


def transform(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply Python-side feature transforms.

    Args:
        df: Raw features loaded from data source.
        config: Loaded config.yaml as dict.

    Returns:
        DataFrame with engineered features. Must preserve row count.
        All feature columns must be numeric.
    """
    df = df.copy()

    # --- Baseline: make raw data model-ready ---

    # Drop columns that are known to be >95% null across the full dataset
    # (hardcoded list ensures consistent schema across train/val/oot splits)
    high_null_cols = [
        "id_07", "id_08", "id_21", "id_22", "id_23", "id_24", "id_25",
        "id_26", "id_27", "dist2",
    ]
    drop_cols = [c for c in high_null_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Encode categorical columns as numeric
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        # Simple frequency encoding for baseline
        freq = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq).fillna(0).astype(float)

    # Fill remaining NaN with -999 (XGBoost handles this natively)
    df = df.fillna(-999)

    return df
