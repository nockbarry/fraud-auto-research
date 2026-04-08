"""Data loading with support for local parquet files or BigQuery.

When config has `local_data.enabled: true`, loads pre-split parquet files from disk.
Otherwise, uses BigQuery with SQL parameterization and caching.
"""

import sys
from pathlib import Path

import pandas as pd

from harness.utils import ROOT_DIR, ensure_cache_dir, file_hash, load_config


# --- Local parquet loading ---

def _load_local(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-split parquet files from local data directory."""
    local_cfg = config["local_data"]
    data_dir = ROOT_DIR / local_cfg.get("data_dir", "data")
    prefix = local_cfg.get("prefix", "raw")

    splits = {}
    for name in ("train", "val", "oot"):
        path = data_dir / f"{prefix}_{name}.parquet"
        print(f"Loading {name} from {path.name}...")
        splits[name] = pd.read_parquet(path)
        print(f"  {len(splits[name]):,} rows, {len(splits[name].columns)} columns")

    return splits["train"], splits["val"], splits["oot"]


# --- BigQuery loading ---

def _read_sql_template() -> str:
    """Read the features.sql template (used only by the dormant BigQuery path)."""
    sql_path = ROOT_DIR / "scripts" / "features.sql"
    return sql_path.read_text()


def _substitute_sql(template: str, config: dict, date_start: str, date_end: str) -> str:
    """Substitute placeholders in the SQL template."""
    bq = config["bigquery"]
    segment = config["segment"]
    fraud = config["fraud_type"]

    date_filter = f"txn_date >= '{date_start}' AND txn_date <= '{date_end}'"
    segment_filter = segment.get("filter_sql", "1=1")

    return template.format(
        project=bq["project"],
        dataset=bq["dataset"],
        source_table=bq["source_table"],
        label_column=fraud["label_column"],
        date_filter=date_filter,
        segment_filter=segment_filter,
    )


def _query_or_cache(sql: str, cache_key: str, config: dict) -> pd.DataFrame:
    """Execute BQ query or load from parquet cache."""
    cache_dir = ensure_cache_dir(config)
    cache_path = cache_dir / f"{cache_key}.parquet"

    if cache_path.exists():
        print(f"  Cache hit: {cache_path.name}")
        return pd.read_parquet(cache_path)

    print(f"  Querying BigQuery ({len(sql)} chars)...")
    from harness.utils import get_bq_client

    client = get_bq_client(config)
    timeout = config.get("execution", {}).get("bq_timeout_seconds", 300)

    df = client.query(sql, timeout=timeout).to_dataframe()

    max_rows = config["bigquery"].get("max_rows")
    if max_rows and len(df) > max_rows:
        print(f"  Warning: truncating {len(df)} rows to max_rows={max_rows}")
        df = df.head(max_rows)

    df.to_parquet(cache_path, index=False)
    print(f"  Cached to {cache_path.name} ({len(df)} rows)")
    return df


def _load_bigquery(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data from BigQuery with SQL parameterization and caching."""
    dates = config["dates"]
    label_col = config["fraud_type"]["label_column"]
    sql_template = _read_sql_template()
    sql_hash = file_hash(ROOT_DIR / "scripts" / "features.sql")
    segment_name = config["segment"]["name"]

    splits = {}
    for split_name, start_key, end_key in [
        ("train", "train_start", "train_end"),
        ("val", "val_start", "val_end"),
        ("oot", "oot_start", "oot_end"),
    ]:
        start = dates[start_key]
        end = dates[end_key]
        cache_key = f"{sql_hash}_{segment_name}_{split_name}_{start}_{end}"
        sql = _substitute_sql(sql_template, config, start, end)
        print(f"Loading {split_name} ({start} to {end})...")
        splits[split_name] = _query_or_cache(sql, cache_key, config)

    df_train, df_val, df_oot = splits["train"], splits["val"], splits["oot"]

    # Rename label column to 'label' for consistency
    for df in [df_train, df_val, df_oot]:
        if label_col in df.columns and label_col != "label":
            df.rename(columns={label_col: "label"}, inplace=True)

    return df_train, df_val, df_oot


# --- Public API ---

def _report_balance(df: pd.DataFrame, label_col: str, split_name: str):
    """Print class balance for a data split."""
    n = len(df)
    pos = df[label_col].sum()
    neg = n - pos
    rate = pos / n if n > 0 else 0
    print(f"  {split_name}: {n:,} rows | {int(pos):,} pos ({rate:.4%}) | {int(neg):,} neg")


def load_data(config: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and split data into train/val/OOT.

    Uses local parquet files if config has `local_data.enabled: true`,
    otherwise uses BigQuery.

    Returns:
        (df_train, df_val, df_oot) DataFrames
    """
    if config is None:
        config = load_config()

    if config.get("local_data", {}).get("enabled", False):
        df_train, df_val, df_oot = _load_local(config)
    else:
        df_train, df_val, df_oot = _load_bigquery(config)

    print("\nClass balance:")
    _report_balance(df_train, "label", "train")
    _report_balance(df_val, "label", "val")
    _report_balance(df_oot, "label", "oot")

    return df_train, df_val, df_oot


def invalidate_cache(config: dict | None = None):
    """Clear all cached parquet files."""
    if config is None:
        config = load_config()
    cache_dir = ensure_cache_dir(config)
    for f in cache_dir.glob("*.parquet"):
        f.unlink()
        print(f"Deleted {f.name}")


if __name__ == "__main__":
    config = load_config()
    if len(sys.argv) > 1 and sys.argv[1] == "--invalidate":
        invalidate_cache(config)
    else:
        df_train, df_val, df_oot = load_data(config)
        print(f"\nTrain shape: {df_train.shape}")
        print(f"Val shape:   {df_val.shape}")
        print(f"OOT shape:   {df_oot.shape}")
