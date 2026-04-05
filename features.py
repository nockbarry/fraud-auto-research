"""Feature transforms applied after BigQuery feature extraction.

This file is edited by the agent. The harness calls transform().
"""

import pandas as pd
import numpy as np


def transform(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply Python-side feature transforms.

    Args:
        df: Raw features from BigQuery (via features.sql).
        config: Loaded config.yaml as dict.

    Returns:
        DataFrame with engineered features. Must preserve row count.
        All feature columns must be numeric.
    """
    # Baseline: pass through all features
    # The agent will add transforms here
    return df
