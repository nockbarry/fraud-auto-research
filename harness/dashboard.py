"""Auto-updating experiment dashboard.

Generates per-dataset plot PNGs and a self-contained HTML page with all data
embedded inline — no server or fetch() required.

Each dataset section shows:
  - Plot PNG (composite, AUPRC val+OOT, AUROC, precision@recall, PSI)
  - Experiment table with all runs: hypothesis, status, key metrics
    Kept rows are highlighted green; discarded are muted.

Usage:
    python3 -m harness.dashboard          # regenerate everything
    python3 -m harness.dashboard --open   # regenerate and open in browser
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from harness.experiment_tracker import EXPERIMENTS_DIR, get_sota, list_datasets, load_history
from harness.plot_results import plot_dataset
from harness.utils import ROOT_DIR

import pandas as pd

REPORTS_DIR = ROOT_DIR / "reports"


def _history_to_df(history: list[dict]) -> pd.DataFrame:
    rows = []
    for exp in history:
        ms = exp.get("metrics_summary", {})
        rows.append({
            "composite": ms.get("composite_score", 0) or 0,
            "auprc_val": ms.get("auprc_val", 0) or 0,
            "auprc": ms.get("auprc", 0) or 0,
            "auroc_val": ms.get("auroc_val", 0) or 0,
            "auroc": ms.get("auroc", 0) or 0,
            "prec@recall": ms.get("precision_at_recall", 0) or 0,
            "psi": ms.get("psi", 0) or 0,
            "n_features": ms.get("n_features", 0) or 0,
            "status": exp.get("status", "discard"),
            "hypothesis": exp.get("hypothesis", ""),
        })
    df = pd.DataFrame(rows)
    df.index = range(1, len(df) + 1)
    return df


def _dataset_summary(history: list[dict], dataset: str) -> dict:
    keeps = [e for e in history if e.get("status") == "keep"]
    discards = [e for e in history if e.get("status") == "discard"]
    sota = get_sota(dataset)

    best_auprc_val = best_auprc_oot = best_composite = baseline_auprc = 0

    if keeps:
        baseline_auprc = keeps[0].get("metrics_summary", {}).get("auprc", 0) or 0
        for k in keeps:
            ms = k.get("metrics_summary", {})
            av = ms.get("auprc_val", 0) or 0
            ao = ms.get("auprc", 0) or 0
            c = ms.get("composite_score", 0) or 0
            best_auprc_val = max(best_auprc_val, av)
            best_auprc_oot = max(best_auprc_oot, ao)
            best_composite = max(best_composite, c)

    improvement_pct = ((best_auprc_oot / baseline_auprc - 1) * 100) if baseline_auprc > 0 else 0

    return {
        "dataset": dataset,
        "total": len(history),
        "kept": len(keeps),
        "discarded": len(discards),
        "best_auprc_val": best_auprc_val,
        "best_auprc_oot": best_auprc_oot,
        "baseline_auprc": baseline_auprc,
        "improvement_pct": improvement_pct,
        "best_composite": best_composite,
        "sota_hypothesis": sota.get("hypothesis", "") if sota else "",
        # Full experiment list for the table
        "experiments": [
            {
                "id": e.get("id", ""),
                "status": e.get("status", "discard"),
                "hypothesis": e.get("hypothesis", ""),
                "auprc_val": (e.get("metrics_summary") or {}).get("auprc_val") or 0,
                "auprc_oot": (e.get("metrics_summary") or {}).get("auprc") or 0,
                "auroc_val": (e.get("metrics_summary") or {}).get("auroc_val") or 0,
                "auroc_oot": (e.get("metrics_summary") or {}).get("auroc") or 0,
                "composite": (e.get("metrics_summary") or {}).get("composite_score") or 0,
                "psi": (e.get("metrics_summary") or {}).get("psi") or 0,
                "n_features": (e.get("metrics_summary") or {}).get("n_features") or 0,
            }
            for e in history
        ],
    }


def update_plots() -> dict[str, str]:
    """Regenerate all per-dataset plot PNGs. Returns {dataset: path}."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ds in list_datasets():
        history = load_history(ds)
        if not history:
            continue
        df = _history_to_df(history)
        out_path = str(REPORTS_DIR / f"plot_{ds}.png")
        plot_dataset(df, ds, out_path)
        paths[ds] = out_path
    return paths


def _image_to_data_uri(path: str) -> str:
    import base64
    try:
        data = Path(path).read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def _fmt(v, decimals=4):
    """Format a float, returning em-dash if zero/None."""
    if not v:
        return "—"
    return f"{v:.{decimals}f}"


def _experiment_table_html(experiments: list[dict]) -> str:
    """Build an HTML table of all experiments for a dataset."""
    rows_html = ""
    for exp in experiments:
        status = exp["status"]
        if status == "keep":
            row_class = "row-keep"
            badge = '<span class="badge badge-keep">KEEP</span>'
        elif status == "discard":
            row_class = "row-discard"
            badge = '<span class="badge badge-discard">discard</span>'
        elif status == "crash":
            row_class = "row-crash"
            badge = '<span class="badge badge-crash">crash</span>'
        elif status == "reject_psi":
            row_class = "row-crash"
            badge = '<span class="badge badge-crash">PSI reject</span>'
        else:
            row_class = "row-discard"
            badge = f'<span class="badge badge-discard">{status}</span>'

        hyp = exp["hypothesis"] or "—"
        exp_id = exp["id"]

        rows_html += f"""
        <tr class="{row_class}">
            <td class="td-id">{exp_id}</td>
            <td class="td-status">{badge}</td>
            <td class="td-hyp" title="{hyp}">{hyp}</td>
            <td class="td-num">{_fmt(exp['auprc_val'])}</td>
            <td class="td-num td-oot">{_fmt(exp['auprc_oot'])}</td>
            <td class="td-num">{_fmt(exp['auroc_val'])}</td>
            <td class="td-num td-oot">{_fmt(exp['auroc_oot'])}</td>
            <td class="td-num">{_fmt(exp['composite'])}</td>
            <td class="td-num">{_fmt(exp['psi'])}</td>
            <td class="td-num">{int(exp['n_features']) if exp['n_features'] else '—'}</td>
        </tr>"""

    return f"""
    <div class="table-wrap">
        <table class="exp-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Status</th>
                    <th>Hypothesis</th>
                    <th>AUPRC<br><span class="th-sub">val</span></th>
                    <th>AUPRC<br><span class="th-sub">OOT</span></th>
                    <th>AUROC<br><span class="th-sub">val</span></th>
                    <th>AUROC<br><span class="th-sub">OOT</span></th>
                    <th>Composite<br><span class="th-sub">val</span></th>
                    <th>PSI</th>
                    <th>Feats</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>"""


def generate_dashboard_html(data: dict) -> str:
    """Generate self-contained HTML with all data and images embedded inline."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORTS_DIR / "dashboard.html"

    datasets = data.get("datasets", {})
    last_updated = data.get("last_updated", "")
    total_exp = sum(d["total"] for d in datasets.values())
    total_keep = sum(d["kept"] for d in datasets.values())
    best_auprc = max((d.get("best_auprc_oot", d.get("best_auprc", 0)) for d in datasets.values()), default=0)

    cards_html = f"""
        <div class="card blue"><div class="label">Datasets</div><div class="value">{len(datasets)}</div></div>
        <div class="card purple"><div class="label">Experiments</div><div class="value">{total_exp}</div></div>
        <div class="card green"><div class="label">Kept</div><div class="value">{total_keep}</div></div>
        <div class="card amber"><div class="label">Best AUPRC (OOT)</div><div class="value">{best_auprc:.4f}</div></div>
    """

    datasets_html = ""
    for name, ds in datasets.items():
        img_path = str(REPORTS_DIR / f"plot_{name}.png")
        img_src = _image_to_data_uri(img_path)
        img_tag = f'<img src="{img_src}" alt="{name} results">' if img_src else ""

        table_html = _experiment_table_html(ds.get("experiments", []))

        imp_pct = ds.get("improvement_pct", 0)
        imp_color = "num" if imp_pct >= 0 else "neg"

        datasets_html += f"""
        <div class="dataset-section">
            <h2>{name}</h2>
            <div class="subtitle">{ds.get("sota_hypothesis", "")}</div>
            <div class="stats-row">
                <div class="stat">Experiments: <strong>{ds["total"]}</strong></div>
                <div class="stat">Kept: <span class="num">{ds["kept"]}</span></div>
                <div class="stat">Discarded: <strong>{ds["discarded"]}</strong></div>
                <div class="stat">Best AUPRC (val): <span class="num">{ds.get("best_auprc_val", 0):.4f}</span></div>
                <div class="stat">Best AUPRC (OOT): <span class="num">{ds.get("best_auprc_oot", 0):.4f}</span></div>
                <div class="stat">Baseline (OOT): <strong>{ds.get("baseline_auprc", 0):.4f}</strong></div>
                <div class="stat">Improvement: <span class="{imp_color}">{imp_pct:+.1f}%</span></div>
            </div>
            <div class="ds-body">
                <div class="ds-plot">{img_tag}</div>
                <div class="ds-table">{table_html}</div>
            </div>
        </div>
        """

    ts_display = last_updated.replace("T", " ")[:19] if last_updated else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="30">
    <title>Fraud Auto-Research Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; }}
        .header {{ background: #1e293b; padding: 20px 32px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }}
        .header h1 {{ font-size: 22px; font-weight: 600; }}
        .header .meta {{ font-size: 13px; color: #94a3b8; }}
        .header .live {{ color: #22c55e; font-size: 12px; font-weight: 600; }}
        .container {{ max-width: 1600px; margin: 0 auto; padding: 24px; }}
        .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .card {{ background: #1e293b; border-radius: 10px; padding: 16px; text-align: center; }}
        .card .value {{ font-size: 28px; font-weight: 700; margin: 4px 0; }}
        .card .label {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
        .card.green .value {{ color: #22c55e; }}
        .card.blue .value {{ color: #3b82f6; }}
        .card.amber .value {{ color: #f59e0b; }}
        .card.purple .value {{ color: #a78bfa; }}

        .dataset-section {{ background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 28px; }}
        .dataset-section h2 {{ font-size: 18px; margin-bottom: 4px; }}
        .dataset-section .subtitle {{ font-size: 13px; color: #94a3b8; margin-bottom: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .stats-row {{ display: flex; gap: 20px; margin-bottom: 16px; flex-wrap: wrap; font-size: 13px; }}
        .stat strong {{ color: #f8fafc; }}
        .stat .num {{ color: #22c55e; font-weight: 600; }}
        .stat .neg {{ color: #f87171; font-weight: 600; }}

        /* Plot + table side-by-side */
        .ds-body {{ display: flex; gap: 20px; align-items: flex-start; }}
        .ds-plot {{ flex: 0 0 55%; min-width: 0; }}
        .ds-plot img {{ width: 100%; border-radius: 8px; background: white; display: block; }}
        .ds-table {{ flex: 1 1 0; min-width: 0; overflow-x: auto; }}

        /* Experiment table */
        .table-wrap {{ overflow-x: auto; }}
        .exp-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
        .exp-table thead tr {{ background: #0f172a; position: sticky; top: 0; z-index: 1; }}
        .exp-table th {{ padding: 8px 6px; text-align: center; color: #94a3b8; font-weight: 600; font-size: 11px; border-bottom: 1px solid #334155; white-space: nowrap; }}
        .th-sub {{ font-size: 9px; color: #64748b; font-weight: 400; }}
        .exp-table td {{ padding: 6px 6px; border-bottom: 1px solid #1e2a3a; vertical-align: middle; }}
        .td-id {{ color: #64748b; font-size: 11px; white-space: nowrap; }}
        .td-status {{ text-align: center; white-space: nowrap; }}
        .td-hyp {{ max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #cbd5e1; }}
        .td-num {{ text-align: right; font-variant-numeric: tabular-nums; color: #94a3b8; white-space: nowrap; }}
        .td-oot {{ color: #f59e0b; }}

        /* Status badges */
        .badge {{ display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 10px; font-weight: 700; letter-spacing: 0.4px; text-transform: uppercase; }}
        .badge-keep {{ background: #14532d; color: #4ade80; border: 1px solid #166534; }}
        .badge-discard {{ background: #1c1c1c; color: #64748b; border: 1px solid #334155; }}
        .badge-crash {{ background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }}

        /* Row highlighting */
        .row-keep {{ background: #0d2818; }}
        .row-keep:hover {{ background: #14532d22; }}
        .row-keep .td-num {{ color: #4ade80; }}
        .row-keep .td-oot {{ color: #f59e0b; }}
        .row-keep .td-hyp {{ color: #f0fdf4; }}
        .row-discard {{ background: transparent; }}
        .row-discard:hover {{ background: #1e293b; }}
        .row-crash {{ background: #1a0a0a; opacity: 0.7; }}

        .footer {{ text-align: center; padding: 16px; font-size: 11px; color: #475569; }}

        @media (max-width: 1000px) {{
            .ds-body {{ flex-direction: column; }}
            .ds-plot {{ flex: none; width: 100%; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Fraud Auto-Research Dashboard</h1>
            <div class="meta">Last updated: {ts_display} &nbsp;|&nbsp; Selection on val · OOT is held-out reporting</div>
        </div>
        <span class="live">AUTO-REFRESH 30s</span>
    </div>
    <div class="container">
        <div class="cards">{cards_html}</div>
        <div id="datasets">{datasets_html}</div>
    </div>
    <div class="footer">
        ● val (green line) = drives keep/discard &nbsp;|&nbsp;
        ◆ OOT (amber, one per experiment) = held-out generalization &nbsp;|&nbsp;
        Plots and data regenerated after each experiment.
    </div>
</body>
</html>"""

    html_path.write_text(html)
    return str(html_path)


def update_dashboard():
    """Regenerate plots, build data, write self-contained HTML."""
    update_plots()

    data = {"datasets": {}, "last_updated": datetime.now().isoformat()}
    for ds in list_datasets():
        history = load_history(ds)
        if history:
            data["datasets"][ds] = _dataset_summary(history, ds)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "dashboard_data.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    html_path = generate_dashboard_html(data)
    print(f"  Dashboard updated: {html_path}")


if __name__ == "__main__":
    import sys
    update_dashboard()
    if "--open" in sys.argv:
        import webbrowser
        webbrowser.open(f"file://{REPORTS_DIR}/dashboard.html")
