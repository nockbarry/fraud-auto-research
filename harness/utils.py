"""Shared utilities for the fraud auto-research harness."""

import hashlib
import os
import subprocess
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"
DEFAULT_TSV_PATH = ROOT_DIR / "results.tsv"

TSV_HEADER = "commit\tcomposite\tauprc\tprec@recall\tpsi\tstatus\tdataset\thypothesis"


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml and return as dict."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def get_bq_client(config: dict):
    """Return a BigQuery client for the configured project."""
    from google.cloud import bigquery

    project = config["bigquery"]["project"]
    return bigquery.Client(project=project)


def file_hash(path: str | Path) -> str:
    """SHA-256 hash of file contents. Used for SQL change detection / cache keys."""
    content = Path(path).read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]


def git_short_hash() -> str:
    """Return the current HEAD short hash (7 chars), or 'no-git' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT_DIR,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "no-git"
    except Exception:
        return "no-git"


def append_result(row: dict, tsv_path: str | Path | None = None):
    """Append a result row to results.tsv. Creates the file with header if missing."""
    tsv_path = Path(tsv_path) if tsv_path else DEFAULT_TSV_PATH
    write_header = not tsv_path.exists()

    with open(tsv_path, "a") as f:
        if write_header:
            f.write(TSV_HEADER + "\n")
        line = "\t".join(
            str(row.get(col, ""))
            for col in ["commit", "composite", "auprc", "prec@recall", "psi", "status", "dataset", "hypothesis"]
        )
        f.write(line + "\n")


def ensure_cache_dir(config: dict) -> Path:
    """Create and return the cache directory path."""
    cache_dir = ROOT_DIR / config["bigquery"].get("cache_dir", "data_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
