"""Auto-updating experiment dashboard.

Generates per-dataset plot PNGs at fixed paths and a static HTML page that
references them. The HTML never changes — just the images get overwritten
on each iteration.

Usage:
    python3 -m harness.dashboard          # regenerate everything
    python3 -m harness.dashboard --open   # regenerate and open in browser

The HTML lives at: reports/dashboard.html
Plots live at:     reports/plot_{dataset}.png
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
    """Convert experiment tracker history to the DataFrame format plot_dataset expects."""
    rows = []
    for exp in history:
        ms = exp.get("metrics_summary", {})
        rows.append({
            "composite": ms.get("composite_score", 0) or 0,
            "auprc": ms.get("auprc", 0) or 0,
            "prec@recall": ms.get("precision_at_recall", 0) or 0,
            "psi": ms.get("psi", 0) or 0,
            "status": exp.get("status", "discard"),
            "hypothesis": exp.get("hypothesis", ""),
        })
    df = pd.DataFrame(rows)
    df.index = range(1, len(df) + 1)
    return df


def _dataset_summary(history: list[dict], dataset: str) -> dict:
    """Build summary stats for a dataset."""
    keeps = [e for e in history if e.get("status") == "keep"]
    discards = [e for e in history if e.get("status") == "discard"]
    sota = get_sota(dataset)

    best_auprc = 0
    baseline_auprc = 0
    best_composite = 0
    best_prec = 0

    if keeps:
        baseline_auprc = keeps[0].get("metrics_summary", {}).get("auprc", 0) or 0
        for k in keeps:
            ms = k.get("metrics_summary", {})
            a = ms.get("auprc", 0) or 0
            c = ms.get("composite_score", 0) or 0
            p = ms.get("precision_at_recall", 0) or 0
            if a > best_auprc:
                best_auprc = a
            if c > best_composite:
                best_composite = c
            if p > best_prec:
                best_prec = p

    improvement_pct = ((best_auprc / baseline_auprc - 1) * 100) if baseline_auprc > 0 else 0

    return {
        "dataset": dataset,
        "total": len(history),
        "kept": len(keeps),
        "discarded": len(discards),
        "best_auprc": best_auprc,
        "baseline_auprc": baseline_auprc,
        "improvement_pct": improvement_pct,
        "best_composite": best_composite,
        "best_prec": best_prec,
        "sota_hypothesis": sota.get("hypothesis", "") if sota else "",
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


def generate_dashboard_html():
    """Write the static HTML dashboard that references plot images by fixed paths."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORTS_DIR / "dashboard.html"

    # This HTML references images at ./plot_{dataset}.png
    # It auto-refreshes every 30 seconds to pick up new plots
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="30">
    <title>Fraud Auto-Research Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; }
        .header { background: #1e293b; padding: 20px 32px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 22px; font-weight: 600; }
        .header .meta { font-size: 13px; color: #94a3b8; }
        .header .live { color: #22c55e; font-size: 12px; }
        .container { max-width: 1400px; margin: 0 auto; padding: 24px; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .card { background: #1e293b; border-radius: 10px; padding: 16px; text-align: center; }
        .card .value { font-size: 28px; font-weight: 700; margin: 4px 0; }
        .card .label { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
        .card.green .value { color: #22c55e; }
        .card.blue .value { color: #3b82f6; }
        .card.amber .value { color: #f59e0b; }
        .card.purple .value { color: #a78bfa; }
        .dataset-section { background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 24px; }
        .dataset-section h2 { font-size: 18px; margin-bottom: 4px; }
        .dataset-section .subtitle { font-size: 13px; color: #94a3b8; margin-bottom: 16px; }
        .dataset-section img { width: 100%; border-radius: 8px; background: white; }
        .stats-row { display: flex; gap: 24px; margin-bottom: 16px; flex-wrap: wrap; }
        .stat { font-size: 13px; }
        .stat strong { color: #f8fafc; }
        .stat .num { color: #22c55e; font-weight: 600; }
        .footer { text-align: center; padding: 16px; font-size: 11px; color: #475569; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Fraud Auto-Research Dashboard</h1>
            <span class="meta" id="timestamp"></span>
        </div>
        <span class="live">AUTO-REFRESH 30s</span>
    </div>
    <div class="container">
        <div class="cards" id="summary-cards"></div>
        <div id="datasets"></div>
    </div>
    <div class="footer">Plots auto-regenerate after each experiment. Page auto-refreshes every 30 seconds.</div>

    <script>
        // Load summary.json and build the page dynamically
        async function loadDashboard() {
            const ts = new Date().toLocaleString();
            document.getElementById('timestamp').textContent = 'Last refresh: ' + ts;

            try {
                const resp = await fetch('./dashboard_data.json?' + Date.now());
                const data = await resp.json();
                renderCards(data);
                renderDatasets(data);
            } catch(e) {
                // If JSON not available, just show the images
                document.getElementById('datasets').innerHTML = '<p style="color:#94a3b8;text-align:center">Waiting for experiment data...</p>';
            }
        }

        function renderCards(data) {
            const datasets = Object.values(data.datasets || {});
            const totalExp = datasets.reduce((s, d) => s + d.total, 0);
            const totalKeep = datasets.reduce((s, d) => s + d.kept, 0);
            const bestAuprc = Math.max(...datasets.map(d => d.best_auprc || 0));
            const html = `
                <div class="card blue"><div class="label">Datasets</div><div class="value">${datasets.length}</div></div>
                <div class="card purple"><div class="label">Experiments</div><div class="value">${totalExp}</div></div>
                <div class="card green"><div class="label">Kept</div><div class="value">${totalKeep}</div></div>
                <div class="card amber"><div class="label">Best AUPRC</div><div class="value">${bestAuprc.toFixed(4)}</div></div>
            `;
            document.getElementById('summary-cards').innerHTML = html;
        }

        function renderDatasets(data) {
            let html = '';
            for (const [name, ds] of Object.entries(data.datasets || {})) {
                const imgSrc = './plot_' + name + '.png?' + Date.now();
                html += `
                    <div class="dataset-section">
                        <h2>${name}</h2>
                        <div class="subtitle">${ds.sota_hypothesis || ''}</div>
                        <div class="stats-row">
                            <div class="stat">Experiments: <strong>${ds.total}</strong></div>
                            <div class="stat">Kept: <span class="num">${ds.kept}</span></div>
                            <div class="stat">Discarded: <strong>${ds.discarded}</strong></div>
                            <div class="stat">AUPRC: <span class="num">${(ds.best_auprc || 0).toFixed(4)}</span></div>
                            <div class="stat">Baseline: <strong>${(ds.baseline_auprc || 0).toFixed(4)}</strong></div>
                            <div class="stat">Improvement: <span class="num">+${(ds.improvement_pct || 0).toFixed(1)}%</span></div>
                        </div>
                        <img src="${imgSrc}" alt="${name} results" onerror="this.style.display='none'">
                    </div>
                `;
            }
            document.getElementById('datasets').innerHTML = html;
        }

        loadDashboard();
    </script>
</body>
</html>"""

    html_path.write_text(html)
    return str(html_path)


def update_dashboard_data():
    """Write the JSON data file the dashboard HTML reads."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data = {"datasets": {}, "last_updated": datetime.now().isoformat()}
    for ds in list_datasets():
        history = load_history(ds)
        if history:
            data["datasets"][ds] = _dataset_summary(history, ds)

    json_path = REPORTS_DIR / "dashboard_data.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def update_dashboard():
    """Regenerate plots, data, and ensure HTML exists."""
    update_plots()
    update_dashboard_data()

    html_path = REPORTS_DIR / "dashboard.html"
    if not html_path.exists():
        generate_dashboard_html()

    print(f"  Dashboard updated: {REPORTS_DIR}/dashboard.html")


if __name__ == "__main__":
    import sys

    update_dashboard()

    if "--open" in sys.argv:
        import webbrowser
        webbrowser.open(f"file://{REPORTS_DIR}/dashboard.html")
