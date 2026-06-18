"""Generate a self-contained HTML LP results report for professor review.

Reads:
  data/results/summary.csv              — IMDb LP (wide format, includes RANDOM rows)
  data/results_dblp/lp_summary_pc_v1.csv — DBLP pc v1 (long format)
  data/results_dblp/lp_summary_pc_v2.csv — DBLP pc v2 (long format)

Writes:
  reports/results_lp_summary.html  — self-contained HTML with table + embedded bar chart

Usage:
    python -m scripts.reports.visualize_lp_results
"""
import base64
import io

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# ---- Data loading -----------------------------------------------------------

def load_imdb(path):
    """Load IMDb wide-format CSV (skips # comment lines). Returns list of dicts."""
    import io as _io
    with open(path) as f:
        clean = "\n".join(l for l in f if not l.lstrip('"').startswith("#"))
    df = pd.read_csv(_io.StringIO(clean))
    records = []
    for _, row in df.iterrows():
        records.append({
            "dataset": "IMDb",
            "task": row["task"].upper(),
            "variant": row["variant"],
            "is_random": row["variant"] == "RANDOM",
            "auc":  float(row["auc_mean"]),  "auc_std":  float(row["auc_std"]),
            "mrr":  float(row["mrr_mean"]),  "mrr_std":  float(row["mrr_std"]),
            "h1":   float(row["hits@1_mean"]), "h1_std": float(row["hits@1_std"]),
            "h3":   float(row["hits@3_mean"]), "h3_std": float(row["hits@3_std"]),
        })
    return records


def load_dblp_long(path, variant_label):
    """Load DBLP long-format CSV; returns a single record dict."""
    df = pd.read_csv(path)
    m = df.set_index("metric")["mean"].to_dict()
    s = df.set_index("metric")["std"].to_dict()
    return {
        "dataset": "DBLP",
        "task": "PC",
        "variant": variant_label,
        "is_random": False,
        "auc":  m.get("auc", float("nan")),  "auc_std":  s.get("auc", 0),
        "mrr":  m.get("mrr", float("nan")),  "mrr_std":  s.get("mrr", 0),
        "h1":   m.get("hits@1", float("nan")), "h1_std": s.get("hits@1", 0),
        "h3":   m.get("hits@3", float("nan")), "h3_std": s.get("hits@3", 0),
    }


def dblp_random():
    # 4 candidates (1 pos + ≤3 neg): H@1=0.25, MRR=(1+0.5+0.33+0.25)/4≈0.52, AUC=0.50
    return {
        "dataset": "DBLP", "task": "PC", "variant": "RANDOM", "is_random": True,
        "auc": 0.50, "auc_std": 0.0,
        "mrr": 0.521, "mrr_std": 0.0,
        "h1": 0.25, "h1_std": 0.0,
        "h3": 0.75, "h3_std": 0.0,
        "h10": float("nan"), "h10_std": 0.0,
    }


def load_fb15k(path):
    """Load FB15k-237 wide summary CSV (rows: per_seed / mean / std, col _type)."""
    df = pd.read_csv(path)
    mean_row = df[df["_type"] == "mean"].iloc[0]
    std_row  = df[df["_type"] == "std"].iloc[0]
    def g(row, col):
        return float(row[col]) if col in row.index else float("nan")
    return {
        "dataset": "FB15k",
        "task": "KG",
        "variant": "p1+c2",
        "is_random": False,
        "auc":  float("nan"), "auc_std":  0.0,
        "mrr":  g(mean_row, "mrr"),  "mrr_std":  g(std_row, "mrr"),
        "h1":   g(mean_row, "hits@1"), "h1_std": g(std_row, "hits@1"),
        "h3":   g(mean_row, "hits@3"), "h3_std": g(std_row, "hits@3"),
        "h10":  g(mean_row, "hits@10"), "h10_std": g(std_row, "hits@10"),
    }


# ---- Chart ------------------------------------------------------------------

TASK_LABELS = {"MD": "IMDb\nmd", "MG": "IMDb\nmg", "ML": "IMDb\nml",
               "PC": "DBLP\npc", "KG": "FB15k-237\n(p1+c2)"}
TASK_ORDER  = ["MD", "MG", "ML", "PC", "KG"]

# variant display order per task
VARIANT_ORDER = {
    "MD": ["RANDOM", "v1", "v3"],
    "MG": ["RANDOM", "v1", "v2", "v3", "v4"],
    "ML": ["RANDOM", "v1", "v2", "v3", "v4"],
    "PC": ["RANDOM", "v1", "v2"],
    "KG": ["p1+c2"],
}

PALETTE = {
    "RANDOM": "#c0c0c0",
    # IMDb shades — warm tones
    "v1_IMDb": "#e07b54", "v2_IMDb": "#d45f30",
    "v3_IMDb": "#c44520", "v4_IMDb": "#a83010",
    # DBLP shades — cool tones
    "v1_DBLP": "#5b9bd5", "v2_DBLP": "#2c6fad",
    # FB15k — muted green
    "p1+c2_FB15k": "#6aaa6a",
}


def get_color(variant, dataset):
    if variant == "RANDOM":
        return PALETTE["RANDOM"]
    return PALETTE.get(f"{variant}_{dataset}", "#888888")


def build_chart(records):
    records_by_task = {}
    for r in records:
        records_by_task.setdefault(r["task"], []).append(r)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#fafafa")

    for ax_idx, (metric_key, metric_std_key, metric_label) in enumerate([
        ("mrr", "mrr_std", "MRR"),
        ("h1",  "h1_std",  "Hits@1"),
    ]):
        ax = axes[ax_idx]
        ax.set_facecolor("#f5f5f5")
        ax.spines[["top", "right"]].set_visible(False)

        group_positions = []
        current_x = 0.0
        group_gap = 0.35

        for task in TASK_ORDER:
            if task not in records_by_task:
                continue
            variants_in_order = VARIANT_ORDER[task]
            task_recs = {r["variant"]: r for r in records_by_task[task]}

            bar_w = 0.28
            bar_gap = 0.04
            n = len(variants_in_order)
            group_width = n * (bar_w + bar_gap)
            xs = [current_x + i * (bar_w + bar_gap) for i in range(n)]
            group_center = current_x + group_width / 2

            for i, variant in enumerate(variants_in_order):
                if variant not in task_recs:
                    continue
                r = task_recs[variant]
                val = r[metric_key]
                err = r[metric_std_key]
                color = get_color(variant, r["dataset"])
                hatch = "///" if r["is_random"] else None
                bar = ax.bar(xs[i], val, width=bar_w, color=color, hatch=hatch,
                             edgecolor="white", linewidth=0.8, zorder=3)
                if err > 0.002:
                    ax.errorbar(xs[i], val, yerr=err, fmt="none", color="#333",
                                capsize=3, linewidth=1.2, zorder=4)

            group_positions.append((group_center, TASK_LABELS[task]))
            current_x += group_width + group_gap

        ax.set_xticks([x for x, _ in group_positions])
        ax.set_xticklabels([lbl for _, lbl in group_positions], fontsize=10)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, color="white", linewidth=1.2, zorder=0)
        ax.set_axisbelow(True)

    # Legend
    legend_items = [
        mpatches.Patch(color=PALETTE["RANDOM"], hatch="///", label="Random baseline"),
        mpatches.Patch(color=PALETTE["v1_IMDb"], label="IMDb v1"),
        mpatches.Patch(color=PALETTE["v3_IMDb"], label="IMDb v3"),
        mpatches.Patch(color=PALETTE["v2_IMDb"], label="IMDb v2/v4"),
        mpatches.Patch(color=PALETTE["v1_DBLP"], label="DBLP v1 (area-paper)"),
        mpatches.Patch(color=PALETTE["v2_DBLP"], label="DBLP v2 (area-venue)"),
        mpatches.Patch(color=PALETTE["p1+c2_FB15k"], label="FB15k p1+c2 (no p3)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("DHN Link Prediction — Vanilla Patterns · No Input Features · 3 Seeds\n"
                 "IMDb/DBLP: p1,c2,p3 · FB15k-237: p1,c2 only (p3 infeasible)",
                 fontsize=11, y=1.02)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#fafafa")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ---- HTML table -------------------------------------------------------------

METRIC_COLS = [
    ("auc", "auc_std", "AUC"),
    ("mrr", "mrr_std", "MRR"),
    ("h1",  "h1_std",  "Hits@1"),
    ("h3",  "h3_std",  "Hits@3"),
    ("h10", "h10_std", "Hits@10"),
]


def fmt(val, std, bold=False):
    if np.isnan(val):
        return "—"
    s = f"{val:.3f}"
    if std > 0.0005:
        s += f" <span style='color:#888;font-size:0.85em'>±{std:.3f}</span>"
    return f"<b>{s}</b>" if bold else s


def build_table(records):
    rows = []
    prev_task = None
    for r in records:
        task_key = (r["dataset"], r["task"])
        sep = prev_task != task_key
        prev_task = task_key

        bg = "#f0f4fb" if r["dataset"] == "DBLP" else ("#fff" if not r["is_random"] else "#f7f7f7")
        border_top = "border-top:2px solid #ccc;" if sep else ""
        variant_label = r["variant"]
        if r["is_random"]:
            variant_label = f'<em style="color:#888">random</em>'

        cells = [
            f'<td style="font-weight:600">{r["dataset"]}</td>',
            f'<td>{r["task"]}</td>',
            f'<td>{variant_label}</td>',
        ]
        for mk, sk, _ in METRIC_COLS:
            bold = not r["is_random"]
            cells.append(f'<td style="text-align:right">{fmt(r[mk], r[sk], bold)}</td>')
        rows.append(
            f'<tr style="background:{bg};{border_top}">{"".join(cells)}</tr>'
        )
    return "\n".join(rows)


# ---- HTML assembly ----------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DHN LP Results</title>
<style>
  body {{ font-family: 'Georgia', serif; max-width: 1000px; margin: 40px auto;
         background: #fefefe; color: #222; padding: 0 20px; }}
  h1 {{ font-size: 1.5em; border-bottom: 2px solid #555; padding-bottom: 6px; }}
  h2 {{ font-size: 1.1em; color: #444; margin-top: 2em; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.93em; margin-top: 1em; }}
  th {{ background: #3b5785; color: white; padding: 7px 10px; text-align: left; }}
  td {{ padding: 6px 10px; }}
  tr:hover {{ background: #eef3fb !important; }}
  .note {{ font-size: 0.85em; color: #666; margin-top: 0.5em; }}
  img {{ max-width: 100%; margin-top: 1em; border: 1px solid #ddd; border-radius: 4px; }}
</style>
</head>
<body>
<h1>DHN Link Prediction — Vanilla Patterns</h1>
<p class="note">Patterns: p1, c2, p3 for IMDb &amp; DBLP; p1, c2 only for FB15k-237 (p3 = 188M rows, infeasible).
No input features — <code>nn.Embedding</code> only. 3 seeds.</p>

<h2>Bar Chart: MRR and Hits@1 by Task</h2>
<img src="data:image/png;base64,{chart_b64}" alt="LP results chart">

<h2>Full Results Table</h2>
<table>
<thead><tr>
  <th>Dataset</th><th>Task</th><th>Variant</th>
  <th style="text-align:right">AUC</th>
  <th style="text-align:right">MRR</th>
  <th style="text-align:right">Hits@1</th>
  <th style="text-align:right">Hits@3</th>
  <th style="text-align:right">Hits@10</th>
</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>

<p class="note">
  <b>IMDb tasks:</b> md = movie↔director (19 negs), mg = movie↔genre (2 negs), ml = movie↔link (19 negs).
  IMDb random baseline: md/ml MRR≈0.18 H@1=0.05; mg MRR≈0.61 H@1=0.33 (low-neg task, saturated).<br>
  <b>DBLP task:</b> pc = paper↔conference (≤3 negs at test, 4 candidates).
  Random baseline: MRR≈0.52 H@1=0.25.<br>
  <b>Variants:</b> IMDb md v1=M↔A,D,L v3=M↔D,L+L↔A.
  DBLP v1=area-paper edges, v2=area-venue edges.<br>
  <b>FB15k-237:</b> full entity ranking (14,541 entities), filtered MRR/Hits (head+tail avg).
  DistMult decoder. No p3 — hub entities make p3 enumeration 188M rows (~50GB).
  Published DistMult baseline: MRR≈0.24 H@10≈0.42.
</p>
</body>
</html>
"""


def main():
    records = []
    # IMDb and DBLP records don't have h10 — fill with NaN
    imdb_records = load_imdb("data/results/summary.csv")
    for r in imdb_records:
        r.setdefault("h10", float("nan")); r.setdefault("h10_std", 0.0)
    records += imdb_records

    for v in ["v1", "v2"]:
        r = load_dblp_long(f"data/results_dblp/lp_summary_pc_{v}.csv", v)
        r.setdefault("h10", float("nan")); r.setdefault("h10_std", 0.0)
        records.append(r)
    records.append(dblp_random())

    fb15k_path = "data/results_fb15k/lp_summary_fb15k.csv"
    import os
    if os.path.exists(fb15k_path):
        records.append(load_fb15k(fb15k_path))
        print("Loaded FB15k-237 results.")
    else:
        print(f"Note: {fb15k_path} not found — pull from HPC first.")

    # Sort: IMDb → DBLP → FB15k; random first within each task
    def sort_key(r):
        ds_order = {"IMDb": 0, "DBLP": 1, "FB15k": 2}
        task_order = {"MD": 0, "MG": 1, "ML": 2, "PC": 3, "KG": 4}
        var_order = {"RANDOM": -1, "v1": 0, "v2": 1, "v3": 2, "v4": 3, "p1+c2": 0}
        return (ds_order[r["dataset"]], task_order[r["task"]], var_order.get(r["variant"], 99))

    records.sort(key=sort_key)

    chart_b64 = build_chart(records)
    table_rows = build_table(records)

    html = HTML_TEMPLATE.format(chart_b64=chart_b64, table_rows=table_rows)
    out = "reports/results_lp_summary.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"Written → {out}")


if __name__ == "__main__":
    main()
