"""
plot_comparison_figure.py

Generates a two-panel figure comparing alignment-accuracy relationships
across the ECHR (Study 1) and German Credit (Study 2) domains.

Left panel:  ECHR — strong positive correlation (r ≈ 0.85)
Right panel: German Credit — flat / null correlation (r ≈ 0.15)

Reads:
  - ECHR data from Niklas's per_model_summary.csv
  - German Credit data hardcoded from balanced-subset analysis reports

Output:
  report/icml2026/figures/alignment_vs_accuracy_comparison.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ── PATHS ─────────────────────────────────────────────────────────────────────

ECHR_CSV   = "/tmp/ECHR_decision_cue_alignment/cross_model_results/per_model_summary.csv"
OUTPUT     = "report/icml2026/figures/alignment_vs_accuracy_comparison.png"

# ── GERMAN CREDIT DATA (balanced subset, n=600) ───────────────────────────────
# Source: analyze_weights.py balanced reports in analysis/

GERMAN_RECORDS = [
    # model_display, condition, cosine, accuracy
    ("Claude Haiku 4.5", "baseline",              0.503, 0.535),
    ("Claude Haiku 4.5", "org-externalized",     -0.104, 0.540),
    ("Claude Haiku 4.5", "introspective-externalized", 0.360, 0.518),
    ("GPT-5.4-mini",     "baseline",              0.060, 0.483),
    ("GPT-5.4-mini",     "org-externalized",      0.224, 0.497),
    ("GPT-5.4-mini",     "introspective-externalized", 0.219, 0.495),
    ("GPT-5.4-nano",     "baseline",              0.499, 0.442),
    ("GPT-5.4-nano",     "org-externalized",      0.076, 0.483),
    ("GPT-5.4-nano",     "introspective-externalized", -0.256, 0.448),
    ("Grok 4.1 Fast",    "baseline",             -0.229, 0.488),
    ("Grok 4.1 Fast",    "org-externalized",      0.176, 0.525),
    # Grok introspective excluded (degenerate: 99.5% Good)
    ("DeepSeek v3.2",    "baseline",              0.264, 0.525),
    ("DeepSeek v3.2",    "org-externalized",      0.164, 0.515),
    ("DeepSeek v3.2",    "introspective-externalized", -0.183, 0.508),
]

# ── SHARED VISUAL SETTINGS ────────────────────────────────────────────────────

CONDITION_MARKER = {
    "baseline":                   "o",
    "court-externalized":         "s",
    "org-externalized":           "s",
    "introspective-externalized": "^",
}

CONDITION_LABEL = {
    "baseline":                   "baseline",
    "court-externalized":         "externalized",
    "org-externalized":           "externalized",
    "introspective-externalized": "introspective",
}

CONDITION_SIZE = 130

# Model colours — consistent for models appearing in both studies
MODEL_COLORS = {
    "Claude Haiku 4.5": "#1f77b4",   # blue
    "DeepSeek v3.2":    "#ff7f0e",   # orange
    "GPT-5.4":          "#2ca02c",   # green
    "GPT-5.4-mini":     "#d62728",   # red
    "GPT-5.4-nano":     "#9467bd",   # purple
    "Gemma 4 (31B)":    "#8c564b",   # brown
    "Grok 4.1 Fast":    "#e377c2",   # pink
    "Minimax M2.7":     "#7f7f7f",   # grey
    "Mistral Large":    "#bcbd22",   # yellow-green
    "Qwen 3.6 Plus":    "#17becf",   # cyan
}


def get_color(model):
    return MODEL_COLORS.get(model, "#333333")


def plot_panel(ax, df, x_col, y_col, condition_col, model_col,
               title, xlabel, ylabel, normative_line=None,
               normative_label=None):
    """Draw one scatter panel and return Pearson r, p."""

    conditions_present = []
    for cond_key in ["baseline", "court-externalized", "org-externalized",
                     "introspective-externalized"]:
        if cond_key in df[condition_col].values:
            conditions_present.append(cond_key)

    for _, row in df.iterrows():
        cond   = row[condition_col]
        model  = row[model_col]
        color  = get_color(model)
        marker = CONDITION_MARKER.get(cond, "D")
        ax.scatter(
            row[x_col], row[y_col],
            color=color, marker=marker, s=CONDITION_SIZE,
            edgecolors="white", linewidths=0.5, zorder=3,
        )

    x = df[x_col].values
    y = df[y_col].values

    # Trend line
    m, b   = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min() - 0.05, x.max() + 0.05, 200)
    ax.plot(x_line, m * x_line + b, "--", color="grey",
            alpha=0.55, linewidth=1.2, zorder=1)

    r, p = pearsonr(x, y)

    # Reference lines
    ax.axvline(0, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
    if normative_line is not None:
        ax.axhline(normative_line, color="green", linestyle=":",
                   linewidth=0.8, alpha=0.55)
        xl = ax.get_xlim()
        ax.text(xl[0] + 0.02 * (xl[1] - xl[0]),
                normative_line + 0.007,
                normative_label, color="green", fontsize=7, alpha=0.75)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(
        f"{title}\nPearson $r$ = {r:.3f}  ($p$ = {p:.2e})",
        fontsize=9.5,
    )
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)

    return r, p, conditions_present


def make_figure():
    # ── Load ECHR data ────────────────────────────────────────────────────────
    echr = pd.read_csv(ECHR_CSV)
    # rename columns for uniformity
    echr = echr.rename(columns={
        "coef_cosine_similarity": "cosine",
        "llm_accuracy_vs_court":  "accuracy",
        "model_display":          "model",
    })

    # ── Build German Credit DataFrame ─────────────────────────────────────────
    gc = pd.DataFrame(GERMAN_RECORDS, columns=["model", "condition", "cosine", "accuracy"])

    # ── Collect all models for a unified legend ───────────────────────────────
    all_models = sorted(set(echr["model"].tolist() + gc["model"].tolist()))

    # Collect all conditions (unified labels)
    all_cond_labels = {}
    for cond in (list(echr["condition"].unique()) + list(gc["condition"].unique())):
        label = CONDITION_LABEL.get(cond, cond)
        if label not in all_cond_labels:
            all_cond_labels[label] = CONDITION_MARKER.get(cond, "D")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(wspace=0.35)

    # Left: ECHR
    plot_panel(
        axes[0], echr,
        x_col="cosine", y_col="accuracy",
        condition_col="condition", model_col="model",
        title="Study 1: ECHR Article 6 Decisions\n($n=10$ models × 3 conditions)",
        xlabel="Policy alignment: cosine similarity of ridge cue weights",
        ylabel="LLM accuracy vs. court decision",
        normative_line=0.715, normative_label="court regression ceiling",
    )

    # Right: German Credit
    plot_panel(
        axes[1], gc,
        x_col="cosine", y_col="accuracy",
        condition_col="condition", model_col="model",
        title="Study 2: German Credit Decisions\n(balanced subset, $n=5$ models × 2–3 conditions)",
        xlabel="Policy alignment: cosine similarity of ridge cue weights",
        ylabel="LLM accuracy vs. ground truth labels",
        normative_line=0.751, normative_label="normative ceiling = 75.1%",
    )

    # ── Shared condition legend (shapes) ─────────────────────────────────────
    cond_handles = [
        plt.scatter([], [], marker=marker, s=90, color="grey",
                    edgecolors="white", linewidths=0.5, label=label)
        for label, marker in all_cond_labels.items()
    ]
    fig.legend(
        handles=cond_handles, title="condition",
        loc="lower center", ncol=len(all_cond_labels),
        fontsize=8, title_fontsize=8,
        framealpha=0.85,
        bbox_to_anchor=(0.5, -0.06),
    )

    # ── Model legend (colours) ────────────────────────────────────────────────
    model_handles = [
        plt.scatter([], [], marker="o", s=90, color=get_color(m),
                    edgecolors="white", linewidths=0.5, label=m)
        for m in all_models
    ]
    fig.legend(
        handles=model_handles, title="model",
        loc="lower center", ncol=5,
        fontsize=7.5, title_fontsize=8,
        framealpha=0.85,
        bbox_to_anchor=(0.5, -0.22),
    )

    plt.suptitle(
        "Cue-utilization alignment vs. output accuracy across organizational domains",
        fontsize=11, y=1.01,
    )

    plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {OUTPUT}")


if __name__ == "__main__":
    make_figure()
