"""
plot_scatter.py

Generates a scatter plot of cue-utilization alignment (cosine similarity)
vs. ground truth accuracy for all models and conditions.

Reads analysis report .txt files from the analysis/ directory and parses
the key metrics. Re-run this script whenever new condition results are added.

Output:
    analysis/alignment_vs_accuracy_german_credit.png

Usage:
    python plot_scatter.py
    python plot_scatter.py --output analysis/my_plot.png
    python plot_scatter.py --reports-dir analysis/
"""

import os
import re
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ── REPORT PARSER ──────────────────────────────────────────────────────────────

def parse_report(path: str) -> dict | None:
    """
    Parses key metrics from an analyze_weights.py report .txt file.
    Returns None if the file cannot be parsed.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    def extract(pattern, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    model     = extract(r"Model\s+:\s+(.+)",     str)
    condition = extract(r"Condition\s+:\s+(.+)", str)

    if model is None or condition is None:
        return None

    model     = model.strip()
    condition = condition.strip()

    accuracy   = extract(r"Output accuracy\s+([\d.]+)\s+%")
    cosine     = extract(r"Cosine similarity \[PRIMARY\]\s+([-\d.]+)")
    good_rate  = extract(r"Good classification rate\s+([\d.]+)\s+%")
    kappa      = extract(r"Cohen's kappa\s+([-\d.]+)")
    auc        = extract(r"ROC AUC \(cue-driven, CV\)\s+([\d.]+)")

    if accuracy is None or cosine is None:
        return None

    return {
        "model":     model,
        "condition": condition,
        "accuracy":  accuracy / 100.0,   # store as fraction
        "cosine":    cosine,
        "good_rate": good_rate,
        "kappa":     kappa,
        "auc":       auc,
        "path":      path,
    }


# ── DISPLAY NAMES ──────────────────────────────────────────────────────────────

MODEL_DISPLAY = {
    "anthropic/claude-haiku-4-5": "Claude Haiku 4.5",
    "openai/gpt-5.4-mini":        "GPT-5.4-mini",
    "openai/gpt-5.4-nano":        "GPT-5.4-nano",
    "x-ai/grok-4.1-fast":         "Grok 4.1 Fast",
    "deepseek/deepseek-v3.2":     "DeepSeek v3.2",
}

CONDITION_DISPLAY = {
    "scenario":      "baseline",
    "court_extern":  "court-externalized",
    "introspective": "introspective",
}

CONDITION_MARKER = {
    "scenario":      "o",   # circle
    "court_extern":  "s",   # square
    "introspective": "^",   # triangle
}

CONDITION_SIZE = {
    "scenario":      200,
    "court_extern":  200,
    "introspective": 200,
}

# Use a consistent colour per model
MODEL_COLORS = [
    "#1f77b4",   # blue     — Claude Haiku
    "#d62728",   # red      — GPT-5.4-mini
    "#8c564b",   # brown    — GPT-5.4-nano
    "#e377c2",   # pink     — Grok
    "#ff7f0e",   # orange   — DeepSeek
]


# ── PLOT ───────────────────────────────────────────────────────────────────────

def make_plot(records: list[dict], output_path: str):
    df = pd.DataFrame(records)

    # Map to display names
    df["model_display"] = df["model"].map(MODEL_DISPLAY).fillna(df["model"])

    models     = sorted(df["model_display"].unique())
    conditions = [c for c in ["scenario", "court_extern", "introspective"]
                  if c in df["condition"].unique()]

    model_color = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in df.iterrows():
        cond  = row["condition"]
        mdl   = row["model_display"]
        color = model_color.get(mdl, "grey")
        marker = CONDITION_MARKER.get(cond, "D")
        size   = CONDITION_SIZE.get(cond, 150)

        ax.scatter(
            row["cosine"], row["accuracy"],
            color=color, marker=marker, s=size,
            edgecolors="white", linewidths=0.5, zorder=3,
        )

    # ── TREND LINE (all points) ────────────────────────────────────────────
    x_all = df["cosine"].values
    y_all = df["accuracy"].values

    if len(x_all) >= 3:
        m, b   = np.polyfit(x_all, y_all, 1)
        x_line = np.linspace(x_all.min() - 0.05, x_all.max() + 0.05, 100)
        ax.plot(x_line, m * x_line + b, "--", color="grey",
                alpha=0.5, linewidth=1.2, zorder=1)

        pr, pp   = pearsonr(x_all, y_all)
        sr, sp   = spearmanr(x_all, y_all)
        subtitle = (f"Pearson r = {pr:.3f} (p = {pp:.2e})  |  "
                    f"Spearman rho = {sr:.3f} (p = {sp:.2e})")
    else:
        subtitle = ""

    # ── REFERENCE LINES ───────────────────────────────────────────────────
    ax.axvline(0, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axhline(0.70, color="green", linestyle=":", linewidth=0.8, alpha=0.5,
               label="_nolegend_")
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > -0.6 else -0.55,
            0.703, "70% base rate", color="green", fontsize=7, alpha=0.7)

    # ── LEGENDS ───────────────────────────────────────────────────────────
    # Condition legend (shape)
    cond_handles = [
        plt.scatter([], [], marker=CONDITION_MARKER[c], s=100, color="grey",
                    edgecolors="white", linewidths=0.5,
                    label=CONDITION_DISPLAY.get(c, c))
        for c in conditions
        if c in CONDITION_MARKER
    ]
    legend1 = ax.legend(
        handles=cond_handles, title="condition",
        loc="upper left", fontsize=8, title_fontsize=8,
        framealpha=0.85,
    )
    ax.add_artist(legend1)

    # Model legend (colour)
    model_handles = [
        plt.scatter([], [], marker="o", s=100, color=model_color[m],
                    edgecolors="white", linewidths=0.5, label=m)
        for m in models
    ]
    ax.legend(
        handles=model_handles, title="model",
        loc="lower right", fontsize=8, title_fontsize=8,
        framealpha=0.85,
    )

    # ── LABELS ────────────────────────────────────────────────────────────
    ax.set_xlabel("Policy alignment: cosine similarity of ridge cue weights",
                  fontsize=10)
    ax.set_ylabel("Accuracy (vs. ground truth labels)", fontsize=10)
    ax.set_title(
        "Cue-utilization alignment vs. ground truth accuracy (German Credit)",
        fontsize=11,
    )
    if subtitle:
        ax.set_title(
            "Cue-utilization alignment vs. ground truth accuracy (German Credit)\n"
            + subtitle,
            fontsize=10,
        )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved: {output_path}")
    print(f"  {len(df)} data points  |  {len(models)} models  |  {len(conditions)} conditions")
    if subtitle:
        print(f"  {subtitle}")


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot cosine similarity vs. ground truth accuracy from analysis reports."
    )
    parser.add_argument("--reports-dir", default="analysis",
                        help="Directory containing report_*.txt files (default: analysis/)")
    parser.add_argument("--output",      default="analysis/alignment_vs_accuracy_german_credit.png",
                        help="Output PNG path")
    parser.add_argument("--conditions",  nargs="*",
                        default=["scenario", "court_extern", "introspective"],
                        help="Which conditions to include (default: all three)")
    parser.add_argument("--balanced",    action="store_true",
                        help="Read balanced reports (report_*_balanced.txt) and save to "
                             "alignment_vs_accuracy_german_credit_balanced.png")
    args = parser.parse_args()

    if args.balanced:
        pattern = "report_results_*_balanced.txt"
        if args.output == "analysis/alignment_vs_accuracy_german_credit.png":
            args.output = "analysis/alignment_vs_accuracy_german_credit_balanced.png"
    else:
        pattern = "report_results_*.txt"
        # Exclude balanced reports from the default run
        pass

    report_files = sorted(glob.glob(os.path.join(args.reports_dir, pattern)))
    if not args.balanced:
        report_files = [f for f in report_files if "_balanced" not in f]

    if not report_files:
        print(f"No report files found in {args.reports_dir}/")
        return

    records = []
    for path in report_files:
        r = parse_report(path)
        if r is None:
            print(f"  Skipping (could not parse): {os.path.basename(path)}")
            continue
        if r["condition"] not in args.conditions:
            continue
        records.append(r)
        print(f"  Loaded: {r['model'].split('/')[-1]:<20} [{r['condition']:<15}] "
              f"cosine={r['cosine']:+.3f}  acc={r['accuracy']*100:.1f}%")

    if not records:
        print("No valid records found.")
        return

    # Deduplicate: keep only the most recent report per (model, condition).
    # Reports are sorted by filename which includes a timestamp, so last wins.
    seen = {}
    for r in records:
        key = (r["model"], r["condition"])
        seen[key] = r   # later file (alphabetically later timestamp) overwrites earlier
    records = list(seen.values())
    print(f"\nAfter deduplication: {len(records)} unique model/condition pairs")

    make_plot(records, args.output)


if __name__ == "__main__":
    main()
