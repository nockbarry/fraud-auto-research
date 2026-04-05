"""Experiment context generator for the autonomous agent.

Produces a structured summary of experiment history, feature importance trends,
what's been tried, what worked, what hasn't been attempted, and recommended
next directions. This is the agent's "memory" between iterations.

Usage:
    python3 -m harness.context ieee-cis
    python3 -m harness.context fraud-sim

    # Programmatic (called after --save):
    from harness.context import generate_context
    ctx = generate_context("ieee-cis")
    print(ctx)
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from harness.experiment_tracker import (
    EXPERIMENTS_DIR,
    get_sota,
    list_datasets,
    load_history,
)


# All known recipe/technique categories and their keywords
TECHNIQUE_CATEGORIES = {
    "velocity": ["velocity", "vel_", "gap", "burst", "daily_rate", "time_since"],
    "behavioral": ["behav_", "deviation", "zscore", "profile", "hour_deviation"],
    "target_encoding": ["target_enc", "_te", "smoothing", "min_samples", "oof"],
    "interaction_te": ["interaction", "card_x_", "card1_x_", "_x_"],
    "identity": ["identity", "modal", "match", "profile_match", "consistency"],
    "entity_sharing": ["entity", "shared_", "n_cards_per", "sharing"],
    "amount_patterns": ["amt_", "round", "cents", "decimal", "corridor", "iqr"],
    "geo_features": ["geo_", "distance", "lat", "long", "city_pop"],
    "time_features": ["hour", "day_of_week", "weekend", "cyclical", "night"],
    "anomaly": ["anomaly", "mahalanobis", "isolation", "outlier"],
    "model_tuning": ["depth", "learning_rate", "n_estimators", "trees", "ensemble", "subsample", "colsample"],
    "class_weight": ["weight", "scale_pos", "focal", "imbalance", "undersamp"],
    "feature_selection": ["drop", "remove", "select", "prune", "importance"],
    "aggregation": ["agg", "count", "mean", "std", "per_card", "per_merchant"],
}


def _categorize_experiment(exp: dict) -> list[str]:
    """Determine which technique categories an experiment touched."""
    hyp = (exp.get("hypothesis", "") or "").lower()
    categories = []
    for cat, keywords in TECHNIQUE_CATEGORIES.items():
        if any(kw in hyp for kw in keywords):
            categories.append(cat)
    return categories or ["other"]


def _feature_importance_trend(history: list[dict], n_recent: int = 5) -> dict:
    """Track how top feature importances change across recent keeps."""
    keeps = [e for e in history if e.get("status") == "keep"]
    recent = keeps[-n_recent:]

    if not recent:
        return {}

    # Collect all features mentioned across recent experiments
    all_features = set()
    for exp in recent:
        top = exp.get("top_features", {})
        all_features.update(top.keys())

    # Build trend per feature
    trends = {}
    for feat in all_features:
        values = []
        for exp in recent:
            top = exp.get("top_features", {})
            values.append(top.get(feat, 0))
        avg = sum(values) / len(values)
        latest = values[-1]
        first = values[0]
        if first > 0:
            direction = "growing" if latest > first * 1.1 else "declining" if latest < first * 0.9 else "stable"
        else:
            direction = "new" if latest > 0 else "absent"
        trends[feat] = {
            "latest": latest,
            "avg": avg,
            "direction": direction,
        }

    return dict(sorted(trends.items(), key=lambda x: -x[1]["latest"])[:15])


def _identify_untried(history: list[dict], dataset: str) -> list[str]:
    """Identify technique categories that haven't been attempted."""
    tried = set()
    for exp in history:
        tried.update(_categorize_experiment(exp))

    all_categories = set(TECHNIQUE_CATEGORIES.keys())
    untried = all_categories - tried

    # Filter by dataset relevance
    if dataset == "fraud-sim":
        untried.discard("identity")  # no identity columns
    if dataset == "ieee-cis":
        untried.discard("geo_features")  # no geo data

    return sorted(untried)


def _streak_analysis(history: list[dict]) -> dict:
    """Analyze recent keep/discard streaks."""
    if not history:
        return {"current_streak": "none", "streak_length": 0}

    recent = history[-10:]
    statuses = [e.get("status", "") for e in recent]

    # Current streak
    current = statuses[-1]
    length = 1
    for s in reversed(statuses[:-1]):
        if s == current:
            length += 1
        else:
            break

    return {
        "current_streak": current,
        "streak_length": length,
        "last_10": f"{sum(1 for s in statuses if s == 'keep')} kept, {sum(1 for s in statuses if s == 'discard')} discarded",
    }


def generate_context(dataset: str) -> str:
    """Generate the full experiment context string for the agent."""
    history = load_history(dataset)
    sota = get_sota(dataset)

    if not history:
        return f"No experiments yet for {dataset}. Run a baseline first."

    keeps = [e for e in history if e.get("status") == "keep"]
    discards = [e for e in history if e.get("status") == "discard"]

    lines = []
    lines.append(f"EXPERIMENT CONTEXT: {dataset}")
    lines.append(f"{'='*60}")

    # SOTA summary
    if sota:
        ms = sota.get("metrics_summary", {})
        lines.append(f"\nSOTA: {sota['id']} — AUPRC={ms.get('auprc', '?'):.4f}, Composite={ms.get('composite_score', '?'):.4f}")
        lines.append(f"  Hypothesis: {sota.get('hypothesis', '?')}")
        ci = ms.get("auprc_ci")
        if ci:
            lines.append(f"  AUPRC 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        lines.append(f"  Features: {ms.get('n_features', '?')}")

    # Top features from SOTA
    if sota and sota.get("top_features"):
        lines.append(f"\nTop features (current SOTA):")
        for feat, imp in list(sota["top_features"].items())[:10]:
            lines.append(f"  {feat:<45} {imp:.4f}")

    # Experiment summary
    lines.append(f"\nExperiment history: {len(history)} total | {len(keeps)} kept | {len(discards)} discarded")

    baseline = keeps[0] if keeps else None
    if baseline and sota:
        baseline_auprc = baseline.get("metrics_summary", {}).get("auprc", 0) or 0
        sota_auprc = sota.get("metrics_summary", {}).get("auprc", 0) or 0
        if baseline_auprc > 0:
            pct = (sota_auprc / baseline_auprc - 1) * 100
            lines.append(f"  Improvement: {baseline_auprc:.4f} -> {sota_auprc:.4f} ({pct:+.1f}%)")

    # Recent experiments
    lines.append(f"\nLast 10 experiments:")
    for exp in history[-10:]:
        ms = exp.get("metrics_summary", {})
        status = exp.get("status", "?")
        marker = "+" if status == "keep" else "-" if status == "discard" else "!"
        auprc = ms.get("auprc", 0) or 0
        hyp = (exp.get("hypothesis", "") or "")[:65]
        lines.append(f"  [{marker}] {exp['id']}: AUPRC={auprc:.4f} — {hyp}")

    # Technique analysis
    lines.append(f"\nTechnique success rates:")
    cat_results = defaultdict(lambda: {"keep": 0, "discard": 0})
    for exp in history:
        cats = _categorize_experiment(exp)
        for cat in cats:
            cat_results[cat][exp.get("status", "discard")] += 1

    for cat, counts in sorted(cat_results.items(), key=lambda x: -(x[1]["keep"])):
        total = counts["keep"] + counts["discard"]
        rate = counts["keep"] / total * 100 if total > 0 else 0
        lines.append(f"  {cat:<25} {counts['keep']}/{total} kept ({rate:.0f}%)")

    # Untried techniques
    untried = _identify_untried(history, dataset)
    if untried:
        lines.append(f"\nUntried techniques (from recipes.md):")
        for t in untried:
            lines.append(f"  - {t}")

    # Feature importance trends
    trends = _feature_importance_trend(history)
    if trends:
        lines.append(f"\nFeature importance trends (last {min(5, len(keeps))} keeps):")
        for feat, info in list(trends.items())[:10]:
            lines.append(f"  {feat:<45} {info['latest']:.4f} ({info['direction']})")

    # Streak analysis
    streak = _streak_analysis(history)
    if streak["streak_length"] >= 3:
        lines.append(f"\nWarning: {streak['streak_length']}-experiment {streak['current_streak']} streak.")
        if streak["current_streak"] == "discard":
            lines.append("  Consider trying a different category of technique.")

    # Recommendations
    lines.append(f"\nRecommended next steps:")

    if untried:
        lines.append(f"  1. Try untried technique: {untried[0]} (see recipes.md)")
    if trends:
        growing = [f for f, t in trends.items() if t["direction"] == "growing"]
        if growing:
            lines.append(f"  2. Build on growing features: {', '.join(growing[:3])}")
        declining = [f for f, t in trends.items() if t["direction"] == "declining" and t["latest"] > 0.05]
        if declining:
            lines.append(f"  3. Investigate declining high-importance features: {', '.join(declining[:3])}")
    if streak.get("current_streak") == "discard" and streak["streak_length"] >= 3:
        lines.append(f"  4. Break the streak — try a radically different approach")

    return "\n".join(lines)


def print_context(dataset: str):
    """Print the context to stdout."""
    print(generate_context(dataset))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print_context(sys.argv[1])
    else:
        for ds in list_datasets():
            print_context(ds)
            print()
