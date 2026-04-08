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
import re
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
    "geo_distance": ["haversine", "geo_dist", "from_home", "centroid"],
    "time_features": ["hour", "day_of_week", "weekend", "cyclical", "night"],
    "cyclic_time": ["sin_hour", "cos_hour", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "cyclic"],
    "terminal_risk": ["term_fraud", "terminal_risk", "merchant_risk", "term_is_high", "term_log_volume"],
    "anomaly": ["anomaly", "mahalanobis", "isolation", "outlier"],
    "model_tuning": ["depth", "learning_rate", "n_estimators", "trees", "ensemble", "subsample", "colsample"],
    "class_weight": ["weight", "scale_pos", "focal", "imbalance", "undersamp"],
    "feature_selection": ["drop", "remove", "select", "prune", "importance"],
    "aggregation": ["agg", "count", "mean", "std", "per_card", "per_merchant"],
    "uid_construction": ["uid", "uid_", "_uid", "construct_uid", "card1_addr1"],
    "window_velocity": ["vel_1h", "vel_24h", "vel_7d", "window_velocity", "rolling_count", "vel_60s", "vel_600s"],
    "rolling_terminal": ["term_fraud_28d", "rolling_term", "term_compromise", "rolling_fraud_rate"],
}


def _categorize_experiment(exp: dict) -> list[str]:
    """Determine which technique categories an experiment touched.

    Scans both the hypothesis text and the names of features that ended up
    in `top_features`. Feature names are highly diagnostic of which technique
    family was used (e.g. `velocity_burst_count` → velocity).
    """
    hyp = (exp.get("hypothesis", "") or "").lower()
    feature_names = " ".join((exp.get("top_features", {}) or {}).keys()).lower()
    search_text = hyp + " " + feature_names
    categories = []
    for cat, keywords in TECHNIQUE_CATEGORIES.items():
        if any(kw in search_text for kw in keywords):
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


def _load_journal(dataset: str) -> str | None:
    """Load the agent's journal for this dataset, if it exists.

    Truncated at 4 KB to bound context size and force the agent to prune.
    """
    candidates = [
        Path(f"journal_{dataset}.md"),
        Path(f"journal_{dataset.replace('-', '_')}.md"),
        Path(f"journal_{dataset.replace('-cis', '')}.md"),  # ieee-cis -> ieee
        Path(f"journal_{dataset.replace('fraud-', '')}.md"),  # fraud-sim -> sim
    ]
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if p.exists():
            text = p.read_text()
            if len(text) > 4096:
                text = text[:4096] + "\n... [journal truncated at 4 KB — PRUNE IT]"
            return text
    return None


_CAMPAIGN_PAT = re.compile(
    r"([\w-]+)\s*campaign\s+step\s*(\d+)\s*(?:/|of)\s*(\d+)",
    re.IGNORECASE,
)


def _extract_campaigns(history: list[dict]) -> dict:
    """Scan recent hypotheses for `<name> campaign step X/Y` markers.

    Returns: {campaign_name: [(step, total, exp_id, status, ago), ...]}
    """
    campaigns = defaultdict(list)
    recent = history[-15:]
    for i, exp in enumerate(recent):
        hyp = exp.get("hypothesis", "") or ""
        m = _CAMPAIGN_PAT.search(hyp)
        if m:
            name = m.group(1).lower()
            step = int(m.group(2))
            total = int(m.group(3))
            ago = len(recent) - 1 - i
            campaigns[name].append({
                "step": step,
                "total": total,
                "exp_id": exp.get("id", "?"),
                "status": exp.get("status", "?"),
                "ago": ago,
            })
    return dict(campaigns)


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

    # Journal — first thing the agent sees, before metrics distract it
    journal = _load_journal(dataset)
    if journal:
        lines.append("\nAGENT JOURNAL (your own notes — re-read and update before this experiment if stale):")
        lines.append(journal)
        lines.append("")
    else:
        lines.append(
            f"\nAGENT JOURNAL: <none yet — create journal_{dataset}.md with sections: "
            f"Current Thesis, Active Campaign, Open Questions, Lessons Learned, Discarded Theses>"
        )

    # Per-column univariate analysis — surfaces raw signals the agent might be ignoring
    try:
        from harness.column_analysis import format_for_context, is_stale
        col_block = format_for_context(dataset, max_rows=20)
        if col_block:
            lines.append("")
            lines.append(col_block)
        else:
            lines.append(
                f"\nCOLUMN ANALYSIS: <none yet — run `python3 -m harness.column_analysis {dataset}` "
                f"to compute univariate IV / null-flag AUC for raw columns>"
            )
        # Stale-warning so the agent knows to refresh
        if col_block and is_stale(dataset, len(history), kind="raw"):
            lines.append(
                f"  (raw column analysis is stale — re-run `python3 -m harness.column_analysis "
                f"{dataset} --refresh`)"
            )
    except Exception as e:
        lines.append(f"\nCOLUMN ANALYSIS: <unavailable — {e}>")

    # SOTA summary
    if sota:
        ms = sota.get("metrics_summary", {})
        auprc_val = ms.get("auprc_val") or ms.get("auprc_val", ms.get("auprc"))
        auprc_oot = ms.get("auprc")
        composite = ms.get("composite_score")
        lines.append(f"\nSOTA: {sota['id']} — AUPRC_val={auprc_val:.4f} | AUPRC_oot={auprc_oot:.4f} | Composite(val)={composite:.4f}")
        lines.append(f"  Hypothesis: {sota.get('hypothesis', '?')}")
        val_ci = ms.get("auprc_val_ci")
        oot_ci = ms.get("auprc_ci")
        if val_ci:
            lines.append(f"  AUPRC_val 95% CI: [{val_ci[0]:.4f}, {val_ci[1]:.4f}]")
        if oot_ci:
            lines.append(f"  AUPRC_oot 95% CI: [{oot_ci[0]:.4f}, {oot_ci[1]:.4f}]")
        lines.append(f"  Features: {ms.get('n_features', '?')}")
        # Overfitting signals
        auroc_gap = ms.get("auroc_train_val_gap")
        tv_psi = ms.get("train_val_psi")
        ci_w = ms.get("ci_width_val")
        overfit_warnings = []
        if auroc_gap is not None and auroc_gap > 0.03:
            severity = "HIGH" if auroc_gap > 0.10 else "moderate"
            overfit_warnings.append(f"AUROC gap train/val={auroc_gap:.4f} ({severity} overfit risk)")
        if tv_psi is not None and tv_psi > 0.10:
            severity = "HIGH" if tv_psi > 0.15 else "moderate"
            overfit_warnings.append(f"Score PSI train→val={tv_psi:.4f} ({severity} score drift)")
        if ci_w is not None and ci_w > 0.02:
            overfit_warnings.append(f"Val CI width={ci_w:.4f} (wide CI — OOT may regress)")
        if overfit_warnings:
            lines.append(f"  Overfit warnings (current SOTA):")
            for w in overfit_warnings:
                lines.append(f"    ! {w}")

    # Top features from SOTA
    if sota and sota.get("top_features"):
        lines.append(f"\nTop features (current SOTA):")
        for feat, imp in list(sota["top_features"].items())[:10]:
            lines.append(f"  {feat:<45} {imp:.4f}")

    # High-PSI features from SOTA (val→OOT instability per feature)
    if sota and sota.get("feature_psi"):
        lines.append(f"\nUnstable features — val→OOT PSI (current SOTA):")
        for feat, psi_val in sota["feature_psi"].items():
            lines.append(f"  {feat:<45} PSI={psi_val:.4f} — drifts val→OOT, consider stabilizing or removing")

    # High-PSI features from SOTA (train→val instability per feature)
    if sota and sota.get("feature_train_val_psi"):
        lines.append(f"\nUnstable features — train→val PSI (current SOTA):")
        for feat, psi_val in sota["feature_train_val_psi"].items():
            lines.append(f"  {feat:<45} PSI={psi_val:.4f} — already drifting train→val, will be worse OOT")

    # Experiment summary
    lines.append(f"\nExperiment history: {len(history)} total | {len(keeps)} kept | {len(discards)} discarded")

    baseline = keeps[0] if keeps else None
    if baseline and sota:
        ms_b = baseline.get("metrics_summary", {})
        ms_s = sota.get("metrics_summary", {})
        # Show both val and OOT improvement
        for label, key in [("val", "auprc_val"), ("oot", "auprc")]:
            b_val = ms_b.get(key) or 0
            s_val = ms_s.get(key) or 0
            if b_val > 0:
                pct = (s_val / b_val - 1) * 100
                lines.append(f"  Improvement ({label}): {b_val:.4f} -> {s_val:.4f} ({pct:+.1f}%)")

    # Recent experiments — with discard reason synthesis
    sota_composite = (sota.get("metrics_summary", {}).get("composite_score") or 0) if sota else 0
    lines.append(f"\nLast 10 experiments:")
    for exp in history[-10:]:
        ms = exp.get("metrics_summary", {})
        status = exp.get("status", "?")
        marker = "+" if status == "keep" else "-" if status == "discard" else "!"
        auprc = ms.get("auprc", 0) or 0
        hyp = (exp.get("hypothesis", "") or "")[:55]

        # Build failure reason
        reason = ""
        if status == "crash":
            reason = " [CRASH]"
        elif status == "timeout":
            reason = " [TIMEOUT]"
        elif status == "reject_psi":
            psi_val = ms.get("psi", 0) or 0
            reason = f" [PSI reject: {psi_val:.4f}]"
        elif status == "discard" and sota:
            exp_comp = ms.get("composite_score", 0) or 0
            delta = exp_comp - sota_composite
            reason = f" [comp {exp_comp:.4f} vs SOTA {sota_composite:.4f}, Δ{delta:+.4f}]"

        lines.append(f"  [{marker}] {exp['id']}: AUPRC={auprc:.4f}{reason} — {hyp}")

    # Campaign tracking — visible scoreboard so drift is obvious
    campaigns = _extract_campaigns(history)
    if campaigns:
        lines.append(f"\nActive campaign tracking (last 15 experiments):")
        for name, steps in sorted(campaigns.items()):
            steps_sorted = sorted(steps, key=lambda s: s["step"])
            total = max(s["total"] for s in steps)
            done = sum(1 for s in steps if s["status"] == "keep")
            attempted_steps = sorted({s["step"] for s in steps})
            min_ago = min(s["ago"] for s in steps)
            step_summary = ", ".join(
                f"{s['exp_id']}={s['status']}(step {s['step']}/{s['total']})"
                for s in steps_sorted
            )
            lines.append(f"  '{name}' — {done}/{total} kept | steps attempted: {attempted_steps}")
            lines.append(f"      {step_summary}")
            lines.append(f"      last activity: {min_ago} experiments ago")
            if min_ago >= 3:
                lines.append(f"      WARNING: campaign stalled — last 3+ experiments were unrelated")
    else:
        lines.append(
            f"\nActive campaign tracking: <NONE — no `step X/Y` markers in last 15 hypotheses>"
        )
        lines.append(
            f"  Single-tweak mode caps you at +0.005/exp. Start a campaign — see program.md."
        )

    # Technique analysis
    lines.append(f"\nTechnique success rates:")
    cat_results = defaultdict(lambda: {"keep": 0, "discard": 0})
    for exp in history:
        status = exp.get("status", "discard")
        if status not in ("keep", "discard"):
            continue  # skip crash / unknown statuses
        cats = _categorize_experiment(exp)
        for cat in cats:
            cat_results[cat][status] += 1

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

    # Dead features in latest experiment (importance < 0.001 in top-10)
    if history:
        latest = history[-1]
        latest_top = latest.get("top_features", {}) or {}
        if latest_top:
            negligible = [(f, imp) for f, imp in latest_top.items() if imp < 0.001]
            if negligible:
                lines.append(f"\nWarning: Features with negligible importance (<0.001) in latest run:")
                for f, imp in negligible:
                    lines.append(f"  - {f} (importance={imp:.6f}) — consider dropping")
            n_features = latest.get("metrics_summary", {}).get("n_features", 0) or 0
            n_in_top = len(latest_top)
            if n_features > n_in_top:
                lines.append(
                    f"  ({n_features - n_in_top} of {n_features} features fell below the top-{n_in_top} threshold)"
                )

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
