"""Directory-per-experiment tracking backend.

Replaces git-based keep/revert with persistent experiment directories.
Every experiment (kept AND discarded) is preserved with its code, metrics, and state.

Structure:
    experiments/
    ├── {dataset}/
    │   ├── index.jsonl              # append-only timeline of all experiments
    │   ├── sota -> exp_015/         # symlink to current best
    │   ├── exp_000_baseline/
    │   │   ├── features.py          # code snapshot
    │   │   ├── model.py
    │   │   ├── metrics.json         # all metrics + feature importances + CIs
    │   │   ├── state.json           # fitted feature state (deployable artifact)
    │   │   └── metadata.json        # hypothesis, status, timestamp, parent
    │   ├── exp_001_velocity/
    │   │   └── ...
    │   └── ...
    └── summary.json                 # cross-dataset summary
"""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from harness.utils import ROOT_DIR

EXPERIMENTS_DIR = ROOT_DIR / "experiments"


def _sanitize_name(s: str, max_len: int = 40) -> str:
    """Convert hypothesis text to a safe directory name fragment."""
    safe = "".join(c if c.isalnum() or c in "-_ " else "" for c in s)
    return safe.strip().replace(" ", "_")[:max_len].rstrip("_").lower()


def _next_exp_id(dataset: str) -> int:
    """Get the next experiment number for a dataset."""
    ds_dir = EXPERIMENTS_DIR / dataset
    if not ds_dir.exists():
        return 0
    existing = [d.name for d in ds_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")]
    if not existing:
        return 0
    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max(nums) + 1 if nums else 0


def save_experiment(
    dataset: str,
    hypothesis: str,
    status: str,
    metrics: dict,
    features_py_path: str | Path | None = None,
    model_py_path: str | Path | None = None,
    state: dict | None = None,
    parent_exp: str | None = None,
    config_snapshot: dict | None = None,
) -> dict:
    """Save a complete experiment snapshot.

    Args:
        dataset: Dataset name (e.g. "ieee-cis", "fraud-sim")
        hypothesis: What was tried
        status: "keep" or "discard"
        metrics: Full metrics dict from run_evaluation()
        features_py_path: Path to current features.py (will be copied)
        model_py_path: Path to current model.py (will be copied)
        state: Fitted feature state dict (JSON-serializable)
        parent_exp: ID of the parent experiment (what we branched from)
        config_snapshot: Key config values

    Returns:
        Experiment metadata dict
    """
    ds_dir = EXPERIMENTS_DIR / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)

    exp_num = _next_exp_id(dataset)
    name_fragment = _sanitize_name(hypothesis)
    exp_dirname = f"exp_{exp_num:03d}_{name_fragment}" if name_fragment else f"exp_{exp_num:03d}"
    exp_dir = ds_dir / exp_dirname
    exp_dir.mkdir(parents=True, exist_ok=True)

    exp_id = f"exp_{exp_num:03d}"

    # Copy code snapshots
    if features_py_path is None:
        features_py_path = ROOT_DIR / "features.py"
    if model_py_path is None:
        model_py_path = ROOT_DIR / "model.py"

    shutil.copy2(features_py_path, exp_dir / "features.py")
    shutil.copy2(model_py_path, exp_dir / "model.py")

    # Save fitted state
    if state is not None:
        try:
            with open(exp_dir / "state.json", "w") as f:
                json.dump(state, f, indent=2, default=str)
        except (TypeError, ValueError):
            # State not fully serializable — save what we can
            with open(exp_dir / "state.json", "w") as f:
                json.dump({"error": "state not fully serializable"}, f)

    # Clean metrics for JSON (handle NaN, numpy types)
    def _clean(obj):
        if isinstance(obj, float):
            if obj != obj or obj == float("inf") or obj == float("-inf"):
                return None
            return round(obj, 6)
        if isinstance(obj, (int,)):
            return int(obj)
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        return obj

    clean_metrics = _clean(metrics)

    # Save metrics
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(clean_metrics, f, indent=2)

    # Build metadata
    sota = get_sota(dataset)
    metadata = {
        "id": exp_id,
        "dir": exp_dirname,
        "dataset": dataset,
        "hypothesis": hypothesis,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parent_exp": parent_exp or (sota["id"] if sota else None),
        "metrics_summary": {
            # Val-based (drive keep/discard)
            "composite_score": clean_metrics.get("composite_score"),
            "auprc_val": clean_metrics.get("auprc_val"),
            "auroc_val": clean_metrics.get("auroc_val"),
            "precision_at_recall_val": clean_metrics.get("precision_at_recall_val"),
            "auprc_val_ci": clean_metrics.get("auprc_val_ci"),
            # OOT-based (held-out reporting)
            "auprc": clean_metrics.get("auprc"),
            "auroc": clean_metrics.get("auroc"),
            "composite_score_oot": clean_metrics.get("composite_score_oot"),
            "precision_at_recall": clean_metrics.get("precision_at_recall"),
            "psi": clean_metrics.get("psi"),
            "n_features": clean_metrics.get("n_features"),
            "auprc_ci": clean_metrics.get("auprc_ci"),
        },
        "top_features": {k: v for k, v in list(clean_metrics.get("top_features", {}).items())[:10]},
        "leakage_warnings": clean_metrics.get("leakage_warnings", []),
        "config_snapshot": config_snapshot,
    }

    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Append to index.jsonl
    index_path = ds_dir / "index.jsonl"
    with open(index_path, "a") as f:
        f.write(json.dumps(metadata) + "\n")

    # Update sota symlink if this is a keep
    if status == "keep":
        _update_sota(dataset, exp_dirname)

    # Update cross-dataset summary
    _update_summary()

    print(f"  Experiment saved: {exp_dirname} [{status}]")
    if status == "keep":
        print(f"  SOTA updated -> {exp_dirname}")

    return metadata


def _update_sota(dataset: str, exp_dirname: str):
    """Update the sota symlink to point to the new best experiment."""
    ds_dir = EXPERIMENTS_DIR / dataset
    sota_link = ds_dir / "sota"

    # Remove existing symlink
    if sota_link.is_symlink() or sota_link.exists():
        sota_link.unlink()

    # Create relative symlink
    sota_link.symlink_to(exp_dirname)


def get_sota(dataset: str) -> dict | None:
    """Get metadata for the current SOTA experiment."""
    ds_dir = EXPERIMENTS_DIR / dataset
    sota_link = ds_dir / "sota"

    if not sota_link.exists():
        return None

    sota_dir = sota_link.resolve()
    meta_path = sota_dir / "metadata.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        return json.load(f)


def get_sota_code(dataset: str) -> tuple[str, str] | None:
    """Get the features.py and model.py code from the SOTA experiment.

    Returns:
        (features_py_content, model_py_content) or None
    """
    ds_dir = EXPERIMENTS_DIR / dataset
    sota_link = ds_dir / "sota"

    if not sota_link.exists():
        return None

    sota_dir = sota_link.resolve()
    features_path = sota_dir / "features.py"
    model_path = sota_dir / "model.py"

    if not features_path.exists() or not model_path.exists():
        return None

    return (features_path.read_text(), model_path.read_text())


def load_history(dataset: str) -> list[dict]:
    """Load the full experiment history for a dataset."""
    index_path = EXPERIMENTS_DIR / dataset / "index.jsonl"
    if not index_path.exists():
        return []
    with open(index_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def get_experiment(dataset: str, exp_id: str) -> dict | None:
    """Load a specific experiment's metadata."""
    history = load_history(dataset)
    for exp in history:
        if exp["id"] == exp_id:
            return exp
    return None


def list_datasets() -> list[str]:
    """List all datasets that have experiments."""
    if not EXPERIMENTS_DIR.exists():
        return []
    return [d.name for d in EXPERIMENTS_DIR.iterdir()
            if d.is_dir() and (d / "index.jsonl").exists()]


def print_status(dataset: str | None = None):
    """Print experiment status for one or all datasets."""
    datasets = [dataset] if dataset else list_datasets()

    for ds in datasets:
        history = load_history(ds)
        if not history:
            continue

        keeps = [e for e in history if e["status"] == "keep"]
        discards = [e for e in history if e["status"] == "discard"]
        sota = get_sota(ds)

        print(f"\n{'='*60}")
        print(f"  {ds}")
        print(f"{'='*60}")
        print(f"  Experiments: {len(history)} total | {len(keeps)} kept | {len(discards)} discarded")

        if sota:
            auprc = sota.get("metrics_summary", {}).get("auprc", "?")
            composite = sota.get("metrics_summary", {}).get("composite_score", "?")
            print(f"  SOTA: {sota['id']} — AUPRC={auprc}, Composite={composite}")
            print(f"  SOTA hypothesis: {sota.get('hypothesis', '?')}")

        # Show last 5 experiments
        print(f"\n  Last 5 experiments:")
        for exp in history[-5:]:
            status_marker = "+" if exp["status"] == "keep" else "-"
            auprc = exp.get("metrics_summary", {}).get("auprc", "?")
            print(f"    [{status_marker}] {exp['id']}: AUPRC={auprc} — {exp.get('hypothesis', '')[:60]}")


def _update_summary():
    """Update the cross-dataset summary.json."""
    summary = {"datasets": {}, "last_updated": datetime.now(timezone.utc).isoformat()}

    for ds in list_datasets():
        history = load_history(ds)
        keeps = [e for e in history if e["status"] == "keep"]
        sota = get_sota(ds)
        best_auprc = max(
            (e.get("metrics_summary", {}).get("auprc", 0) or 0 for e in keeps),
            default=0
        )
        baseline_auprc = keeps[0].get("metrics_summary", {}).get("auprc", 0) if keeps else 0

        summary["datasets"][ds] = {
            "total_experiments": len(history),
            "kept": len(keeps),
            "discarded": len(history) - len(keeps),
            "best_auprc": best_auprc,
            "baseline_auprc": baseline_auprc,
            "improvement_pct": round((best_auprc / baseline_auprc - 1) * 100, 1) if baseline_auprc > 0 else 0,
            "sota_id": sota["id"] if sota else None,
            "sota_hypothesis": sota.get("hypothesis", "") if sota else "",
        }

    with open(EXPERIMENTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def compare_experiments(dataset: str, exp_a: str, exp_b: str) -> str:
    """Generate a text diff comparison between two experiments."""
    ds_dir = EXPERIMENTS_DIR / dataset

    # Find directories
    dir_a = dir_b = None
    for d in ds_dir.iterdir():
        if d.is_dir() and d.name.startswith(exp_a):
            dir_a = d
        if d.is_dir() and d.name.startswith(exp_b):
            dir_b = d

    if not dir_a or not dir_b:
        return f"Could not find experiments {exp_a} and/or {exp_b}"

    lines = [f"Comparing {exp_a} vs {exp_b}\n"]

    # Compare metrics
    meta_a = json.loads((dir_a / "metadata.json").read_text())
    meta_b = json.loads((dir_b / "metadata.json").read_text())

    lines.append(f"  {exp_a}: AUPRC={meta_a['metrics_summary'].get('auprc')} [{meta_a['status']}] — {meta_a['hypothesis']}")
    lines.append(f"  {exp_b}: AUPRC={meta_b['metrics_summary'].get('auprc')} [{meta_b['status']}] — {meta_b['hypothesis']}")

    # Code diff (simplified — just show if files differ)
    for fname in ["features.py", "model.py"]:
        a_code = (dir_a / fname).read_text() if (dir_a / fname).exists() else ""
        b_code = (dir_b / fname).read_text() if (dir_b / fname).exists() else ""
        if a_code == b_code:
            lines.append(f"  {fname}: identical")
        else:
            a_lines = len(a_code.splitlines())
            b_lines = len(b_code.splitlines())
            lines.append(f"  {fname}: differs ({a_lines} vs {b_lines} lines)")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print_status(sys.argv[1])
    else:
        print_status()
