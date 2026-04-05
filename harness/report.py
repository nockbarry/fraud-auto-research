"""Generate HTML experiment report with per-dataset plots and tables.

Usage:
    python3 -m harness.report
    python3 -m harness.report --open   # generate and open in browser
"""

import argparse
import base64
import html
from datetime import datetime
from pathlib import Path

import pandas as pd

from harness.plot_results import load_results, plot_all
from harness.utils import ROOT_DIR


def _img_to_base64(path: str) -> str:
    """Convert an image file to base64 data URI."""
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:image/png;base64,{b64}"


def _status_badge(status: str) -> str:
    colors = {
        "keep": "#059669", "discard": "#dc2626",
        "crash": "#7c3aed", "reject_psi": "#d97706",
    }
    c = colors.get(status, "#6b7280")
    return f'<span style="background:{c};color:white;padding:2px 8px;border-radius:4px;font-size:12px">{html.escape(status)}</span>'


def _dataset_summary(df: pd.DataFrame, name: str) -> dict:
    """Compute summary stats for a dataset."""
    keeps = df[df["status"] == "keep"]
    discards = df[df["status"] == "discard"]
    best_auprc = pd.to_numeric(keeps["auprc"], errors="coerce").max() if len(keeps) else 0
    baseline_auprc = pd.to_numeric(keeps["auprc"], errors="coerce").iloc[0] if len(keeps) else 0
    best_composite = pd.to_numeric(keeps["composite"], errors="coerce").max() if len(keeps) else 0
    best_prec = pd.to_numeric(keeps["prec@recall"], errors="coerce").max() if len(keeps) else 0
    improvement = best_auprc - baseline_auprc

    return {
        "name": name,
        "total": len(df),
        "kept": len(keeps),
        "discarded": len(discards),
        "crashed": len(df[df["status"].isin(["crash", "reject_psi"])]),
        "best_auprc": best_auprc,
        "baseline_auprc": baseline_auprc,
        "improvement": improvement,
        "best_composite": best_composite,
        "best_precision": best_prec,
    }


def generate_report(tsv_path: Path | None = None) -> str:
    """Generate the HTML report and return the output path."""
    df = load_results(tsv_path)
    report_dir = ROOT_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Generate per-dataset plots
    plot_paths = plot_all(tsv_path, report_dir)

    # Determine datasets
    if "dataset" in df.columns:
        datasets = df["dataset"].dropna().unique().tolist()
    else:
        datasets = ["all"]
        df["dataset"] = "all"

    # Build summaries
    summaries = {}
    for ds in datasets:
        ds_df = df[df["dataset"] == ds]
        summaries[ds] = _dataset_summary(ds_df, ds)

    # Build experiment table rows
    table_rows = ""
    for _, row in df.iterrows():
        hyp = html.escape(str(row.get("hypothesis", "")))
        ds = html.escape(str(row.get("dataset", "")))
        table_rows += f"""<tr>
            <td style="font-family:monospace;font-size:12px">{html.escape(str(row.get('commit', '')))}</td>
            <td>{ds}</td>
            <td>{_status_badge(str(row.get('status', '')))}</td>
            <td><b>{pd.to_numeric(row.get('composite', 0), errors='coerce'):.4f}</b></td>
            <td>{pd.to_numeric(row.get('auprc', 0), errors='coerce'):.4f}</td>
            <td>{pd.to_numeric(row.get('prec@recall', 0), errors='coerce'):.4f}</td>
            <td>{pd.to_numeric(row.get('psi', 0), errors='coerce'):.4f}</td>
            <td style="font-size:13px;max-width:400px">{hyp}</td>
        </tr>"""

    # Build dataset cards
    dataset_cards = ""
    for ds in datasets:
        s = summaries[ds]
        plot_path = plot_paths.get(ds, "")
        img_html = ""
        if plot_path and Path(plot_path).exists():
            img_html = f'<img src="{_img_to_base64(plot_path)}" style="width:100%;border-radius:8px;margin-top:16px">'

        pct_improve = (s["improvement"] / s["baseline_auprc"] * 100) if s["baseline_auprc"] > 0 else 0
        dataset_cards += f"""
        <div style="background:white;border-radius:12px;padding:24px;margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,0.1)">
            <h2 style="margin:0 0 16px 0;color:#1e293b">{html.escape(ds)}</h2>
            <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px">
                <div style="background:#f0fdf4;padding:12px;border-radius:8px;text-align:center">
                    <div style="font-size:24px;font-weight:700;color:#059669">{s['best_auprc']:.4f}</div>
                    <div style="font-size:12px;color:#6b7280">Best AUPRC</div>
                </div>
                <div style="background:#eff6ff;padding:12px;border-radius:8px;text-align:center">
                    <div style="font-size:24px;font-weight:700;color:#2563eb">{s['best_composite']:.4f}</div>
                    <div style="font-size:12px;color:#6b7280">Best Composite</div>
                </div>
                <div style="background:#fefce8;padding:12px;border-radius:8px;text-align:center">
                    <div style="font-size:24px;font-weight:700;color:#d97706">{s['best_precision']:.4f}</div>
                    <div style="font-size:12px;color:#6b7280">Best Prec@Recall</div>
                </div>
                <div style="background:#f0fdf4;padding:12px;border-radius:8px;text-align:center">
                    <div style="font-size:24px;font-weight:700;color:#059669">+{pct_improve:.1f}%</div>
                    <div style="font-size:12px;color:#6b7280">vs Baseline</div>
                </div>
                <div style="background:#f8fafc;padding:12px;border-radius:8px;text-align:center">
                    <div style="font-size:24px;font-weight:700;color:#334155">{s['total']}</div>
                    <div style="font-size:12px;color:#6b7280">{s['kept']} kept / {s['discarded']} disc</div>
                </div>
            </div>
            {img_html}
        </div>
        """

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fraud Auto-Research Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f1f5f9; color: #1e293b; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
        h1 {{ font-size: 28px; margin-bottom: 8px; }}
        .subtitle {{ color: #64748b; margin-bottom: 24px; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ background: #f8fafc; padding: 10px 12px; text-align: left; border-bottom: 2px solid #e2e8f0; font-weight: 600; }}
        td {{ padding: 8px 12px; border-bottom: 1px solid #f1f5f9; }}
        tr:hover {{ background: #f8fafc; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Fraud Auto-Research Report</h1>
    <p class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | {len(df)} experiments across {len(datasets)} dataset(s) | v2 leakage-safe harness</p>

    {dataset_cards}

    <div style="background:white;border-radius:12px;padding:24px;box-shadow:0 1px 3px rgba(0,0,0,0.1)">
        <h2 style="margin:0 0 16px 0">All Experiments</h2>
        <table>
            <thead>
                <tr>
                    <th>Commit</th><th>Dataset</th><th>Status</th><th>Composite</th>
                    <th>AUPRC</th><th>Prec@Recall</th><th>PSI</th><th>Hypothesis</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
    </div>
</div>
</body>
</html>"""

    out_path = report_dir / "report.html"
    out_path.write_text(report_html)
    print(f"Report saved to {out_path}")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=str, default=None)
    parser.add_argument("--open", action="store_true", help="Open in browser")
    args = parser.parse_args()

    path = generate_report(Path(args.tsv) if args.tsv else None)
    if args.open:
        import webbrowser
        webbrowser.open(f"file://{path}")
