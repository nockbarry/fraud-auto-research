"""Multi-dataset JSONL results storage.

Structure:
    results/
    ├── index.json
    ├── {dataset}/
    │   ├── dataset.json
    │   └── {segment}/
    │       ├── experiments.jsonl
    │       └── plots/
    └── ...
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from harness.utils import ROOT_DIR, git_short_hash


RESULTS_DIR = ROOT_DIR / "results"


def _ensure_dirs(dataset: str, segment: str) -> Path:
    """Create results directory structure."""
    seg_dir = RESULTS_DIR / dataset / segment
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "plots").mkdir(exist_ok=True)
    return seg_dir


def _update_index(dataset: str, segment: str, description: str = ""):
    """Update the top-level index.json registry."""
    index_path = RESULTS_DIR / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = {"datasets": {}}

    if dataset not in index["datasets"]:
        index["datasets"][dataset] = {
            "description": description,
            "segments": [],
            "total_experiments": 0,
            "best_auprc": None,
        }

    ds = index["datasets"][dataset]
    if segment not in ds["segments"]:
        ds["segments"].append(segment)

    index_path.write_text(json.dumps(index, indent=2))


def log_experiment(
    dataset: str,
    segment: str,
    hypothesis: str,
    status: str,
    metrics: dict,
    experiment_type: str = "feature_engineering",
    files_changed: list[str] | None = None,
    leakage_warnings: list[str] | None = None,
    config_snapshot: dict | None = None,
    dataset_description: str = "",
) -> dict:
    """Append an experiment result to the JSONL log.

    Args:
        dataset: Dataset name (e.g., "ieee-cis", "creditcard-2023")
        segment: Segment name (e.g., "all_transactions", "new_customers")
        hypothesis: What was tried
        status: "keep", "discard", "crash", "reject_psi"
        metrics: Dict of metric values from run_evaluation()
        experiment_type: "feature_engineering", "model_tuning", "feature_analysis"
        files_changed: List of files modified
        leakage_warnings: Any leakage warnings detected
        config_snapshot: Key config values
        dataset_description: Human-readable dataset description (for index)

    Returns:
        The experiment record dict
    """
    seg_dir = _ensure_dirs(dataset, segment)
    _update_index(dataset, segment, dataset_description)

    jsonl_path = seg_dir / "experiments.jsonl"

    # Count existing experiments for ID
    n_existing = 0
    if jsonl_path.exists():
        n_existing = sum(1 for _ in jsonl_path.open())

    exp_id = f"exp_{n_existing:04d}"

    record = {
        "id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_short_hash(),
        "dataset": dataset,
        "segment": segment,
        "hypothesis": hypothesis,
        "experiment_type": experiment_type,
        "files_changed": files_changed or [],
        "status": status,
        "metrics": {
            "composite_score": metrics.get("composite_score"),
            "auprc_oot": metrics.get("auprc"),
            "auprc_val": metrics.get("auprc_val"),
            "precision_at_recall": metrics.get("precision_at_recall"),
            "psi": metrics.get("psi"),
            "fpr": metrics.get("fpr"),
            "review_burden": metrics.get("review_burden"),
            "n_features": metrics.get("n_features"),
            "training_seconds": metrics.get("training_seconds"),
            "transform_latency_ms": metrics.get("transform_latency_ms"),
        },
        "leakage_warnings": leakage_warnings or [],
        "config_snapshot": config_snapshot,
        "is_sota": status == "keep",
    }

    # Clean NaN/inf for JSON
    def _clean(obj):
        if isinstance(obj, float):
            if obj != obj or obj == float("inf") or obj == float("-inf"):
                return None
            return round(obj, 6)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    record = _clean(record)

    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Update index totals
    _update_index_totals(dataset, segment)

    return record


def _update_index_totals(dataset: str, segment: str):
    """Update experiment counts and best metrics in index."""
    index_path = RESULTS_DIR / "index.json"
    if not index_path.exists():
        return

    index = json.loads(index_path.read_text())
    ds = index["datasets"].get(dataset, {})

    # Count all experiments across segments
    total = 0
    best_auprc = None
    for seg in ds.get("segments", []):
        jsonl_path = RESULTS_DIR / dataset / seg / "experiments.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path) as f:
                for line in f:
                    total += 1
                    exp = json.loads(line)
                    if exp.get("status") == "keep":
                        auprc = exp.get("metrics", {}).get("auprc_oot")
                        if auprc is not None and (best_auprc is None or auprc > best_auprc):
                            best_auprc = auprc

    ds["total_experiments"] = total
    ds["best_auprc"] = best_auprc
    index["datasets"][dataset] = ds
    index_path.write_text(json.dumps(index, indent=2))


def load_experiments(dataset: str, segment: str) -> list[dict]:
    """Load all experiments for a dataset/segment."""
    jsonl_path = RESULTS_DIR / dataset / segment / "experiments.jsonl"
    if not jsonl_path.exists():
        return []
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f]


def get_sota(dataset: str, segment: str) -> dict | None:
    """Get the current SOTA experiment for a dataset/segment."""
    experiments = load_experiments(dataset, segment)
    sota = None
    for exp in experiments:
        if exp.get("status") == "keep":
            sota = exp
    return sota


def print_summary(dataset: str | None = None):
    """Print a summary of all experiments across datasets."""
    index_path = RESULTS_DIR / "index.json"
    if not index_path.exists():
        print("No results yet.")
        return

    index = json.loads(index_path.read_text())
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*70}")

    for ds_name, ds in index["datasets"].items():
        if dataset and ds_name != dataset:
            continue
        print(f"\n  {ds_name}: {ds.get('description', '')}")
        print(f"  Total experiments: {ds.get('total_experiments', 0)}")
        print(f"  Best AUPRC: {ds.get('best_auprc', 'N/A')}")
        print(f"  Segments: {', '.join(ds.get('segments', []))}")

        for seg in ds.get("segments", []):
            experiments = load_experiments(ds_name, seg)
            keeps = [e for e in experiments if e["status"] == "keep"]
            discards = [e for e in experiments if e["status"] == "discard"]
            crashes = [e for e in experiments if e["status"] in ("crash", "reject_psi")]

            if experiments:
                print(f"\n    [{seg}] {len(experiments)} experiments: "
                      f"{len(keeps)} kept, {len(discards)} discarded, {len(crashes)} crashed")
                if keeps:
                    best = max(keeps, key=lambda e: e["metrics"].get("auprc_oot") or 0)
                    print(f"    Best: AUPRC={best['metrics'].get('auprc_oot', 'N/A')} "
                          f"— {best.get('hypothesis', 'N/A')}")
