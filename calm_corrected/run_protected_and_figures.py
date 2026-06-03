"""
calm_corrected.run_protected_and_figures
========================================

From the CORRECTED cue-value coefficients (one-hot + standardized, fixed org
policy), produces:

  1. corrected_german_credit_scatter.png — the German Credit panel of Figure 1,
     redrawn with corrected cosines. (The ECHR/Study-1 left panel is unchanged:
     binary cues, different project; it can be reused as-is.)

  2. protected_attribute_usage.{png,csv} — how much each decision-maker's policy
     relies on the protected attributes (age, personal_status_sex,
     foreign_worker) plus three legitimate-cue anchors, for the organization vs
     each LLM at baseline and under org-externalization.

USAGE METRIC: each coefficient vector is L2-normalized to unit length, so
"usage" of an attribute is the share of total policy magnitude on that
attribute's one-hot columns (see ``core.usage``).

Usage:
    python -m calm_corrected.run_protected_and_figures \\
        --repo . --out-dir analysis/corrected --fig-dir report/icml2026/figures
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

from calm_corrected import core


def _compute(repo):
    """Compute, once, the fixed space and every cell's corrected beta + accuracy."""
    decoded = core.load_decoded(repo)
    space = core.build_fixed_space(repo, decoded)
    betas, accs = {}, {}
    for cell in core.CELLS:
        merged = core.prepare_cell(repo, cell, decoded)
        betas[(cell.model, cell.condition)] = core.corrected_beta(merged, space)
        accs[(cell.model, cell.condition)] = core.accuracy_good(merged)[0]
    return space, betas, accs


def _scatter(space, betas, accs, fig_path):
    rows = []
    for cell in core.CELLS:
        _, marker = core.CONDITION_DISPLAY[cell.condition]
        beta = betas[(cell.model, cell.condition)]
        rows.append(dict(model=cell.model, condition=cell.condition, marker=marker,
                         cos=core.cosine(space.beta_org, beta),
                         acc=accs[(cell.model, cell.condition)] / 100))
    scat = pd.DataFrame(rows)
    x, y = scat["cos"].values, scat["acc"].values
    r_all, p_all = pearsonr(x, y)
    sub = scat[scat.model != "GPT-5.4-nano"]
    r_ex, _ = pearsonr(sub["cos"], sub["acc"])

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for _, row in scat.iterrows():
        ax.scatter(row["cos"], row["acc"] * 100, color=core.COLORS[row["model"]],
                   marker=row["marker"], s=130, edgecolors="white", linewidths=0.5, zorder=3)
    m, bb = np.polyfit(x, y, 1)
    xl = np.linspace(x.min() - 0.05, x.max() + 0.05, 200)
    ax.plot(xl, (m * xl + bb) * 100, "--", color="grey", alpha=0.55, lw=1.2, zorder=1)
    ax.axhline(75.1, ls=":", color="black", alpha=0.5, lw=1)
    ax.text(ax.get_xlim()[1], 75.6, "linear ceiling 75.1%", ha="right", va="bottom", fontsize=8, alpha=0.7)
    ax.axvline(0, color="grey", alpha=0.25, lw=0.8)
    ax.set_xlabel("Policy alignment (corrected cosine similarity)")
    ax.set_ylabel("Output accuracy (%)")
    # The fit annotation keeps BOTH the full r and the excluding-GPT-nano r, so the
    # leverage-driven nature of the correlation is visible (not a headline finding).
    ax.set_title("Study 2: German Credit (corrected)\n"
                 f"$r$ = {r_all:+.2f} (p={p_all:.3f}); {r_ex:+.2f} excluding GPT-5.4-nano",
                 fontsize=10)
    mk = [Line2D([0], [0], marker=mm, color="grey", ls="", ms=8, label=ll) for mm, ll in
          [("o", "baseline"), ("s", "org-ext"), ("^", "introspective")]]
    cl = [Line2D([0], [0], marker="o", color=core.COLORS[k], ls="", ms=8, label=k) for k in core.COLORS]
    ax.legend(handles=mk + cl, fontsize=7, loc="lower right", framealpha=0.9, ncol=2)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    return r_all, p_all, r_ex


def _protected(space, betas, fig_path, csv_path):
    attrs = core.PROTECTED + core.ANCHORS
    cols = space.columns

    base_cells = core.cells_for("baseline")
    ext_cells = core.cells_for("org-ext")
    models = list(base_cells)  # all 5 models present at baseline and org-ext

    # --- per-decision-maker usage CSV ---
    recs = [{"attribute": a, "decision_maker": "Organization",
             "usage": core.usage(space.beta_org, cols, a)} for a in attrs]
    for model in models:
        bb_ = betas[(model, "baseline")]
        be_ = betas[(model, "org-ext")]
        for a in attrs:
            recs.append({"attribute": a, "decision_maker": f"{model} (baseline)",
                         "usage": core.usage(bb_, cols, a)})
            recs.append({"attribute": a, "decision_maker": f"{model} (org-ext)",
                         "usage": core.usage(be_, cols, a)})
    udf = pd.DataFrame(recs)
    udf.to_csv(csv_path, index=False)

    # --- grouped bar figure (means across the 5 models) ---
    def mean_usage(condition, a):
        return np.mean([core.usage(betas[(m, condition)], cols, a) for m in models])

    org_u = [core.usage(space.beta_org, cols, a) for a in attrs]
    base_u = [mean_usage("baseline", a) for a in attrs]
    ext_u = [mean_usage("org-ext", a) for a in attrs]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xpos = np.arange(len(attrs))
    w = 0.26
    ax.bar(xpos - w, org_u, w, label="Organization (historical policy)", color="#2c3e50")
    ax.bar(xpos, base_u, w, label="LLMs, baseline (mean of 5)", color="#7fb3d5")
    ax.bar(xpos + w, ext_u, w, label="LLMs, org-externalized (mean of 5)", color="#d98880")
    ax.axvline(len(core.PROTECTED) - 0.5, color="grey", ls="--", alpha=0.5)
    _ymax = ax.get_ylim()[1]
    ax.text(1, _ymax * 0.96, "PROTECTED attributes", ha="center", fontsize=9, weight="bold", color="#922b21")
    ax.text(4, _ymax * 0.96, "legitimate risk cues", ha="center", fontsize=9, weight="bold", color="#1e6f3a")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["age", "personal\nstatus / sex", "foreign\nworker",
                        "checking\naccount", "duration", "credit\namount"], fontsize=9)
    ax.set_ylabel("Relative policy usage\n(share of unit-norm coefficients)")
    ax.set_title("Protected-attribute usage: the organization uses them, the LLMs largely do not\n"
                 "(and org-externalization does not shift the LLMs toward using them)", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return attrs, org_u, base_u, ext_u


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".", help="Path to repo root")
    ap.add_argument("--out-dir", default="analysis/corrected",
                    help="Directory for numeric outputs (CSV), relative to --repo")
    ap.add_argument("--fig-dir", default="report/icml2026/figures",
                    help="Directory for figures, relative to --repo")
    args = ap.parse_args(argv)

    out_dir = os.path.join(args.repo, args.out_dir)
    fig_dir = os.path.join(args.repo, args.fig_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    scatter_png = os.path.join(fig_dir, "corrected_german_credit_scatter.png")
    usage_png = os.path.join(fig_dir, "protected_attribute_usage.png")
    usage_csv = os.path.join(out_dir, "protected_attribute_usage.csv")

    space, betas, accs = _compute(args.repo)
    r_all, p_all, r_ex = _scatter(space, betas, accs, scatter_png)
    attrs, org_u, base_u, ext_u = _protected(space, betas, usage_png, usage_csv)

    # ----- console summary table -----
    cols = space.columns
    print("PROTECTED-ATTRIBUTE RELATIVE USAGE (share of unit-norm policy magnitude)\n")
    hdr = f"{'attribute':<24}{'ORG':>7}{'LLM base(mean)':>16}{'LLM org-ext(mean)':>19}{'shift':>8}"
    print(hdr)
    print("-" * len(hdr))
    for a, o, b, e in zip(attrs, org_u, base_u, ext_u):
        tag = " (protected)" if a in core.PROTECTED else ""
        print(f"{a:<24}{o:>7.3f}{b:>16.3f}{e:>19.3f}{e - b:>+8.3f}{tag}")

    print("\nPer-model protected-attribute usage (baseline -> org-ext):")
    for model in core.cells_for("baseline"):
        bb_ = betas[(model, "baseline")]
        be_ = betas[(model, "org-ext")]
        parts = [f"{a.split('_')[0]}:{core.usage(bb_, cols, a):.2f}->{core.usage(be_, cols, a):.2f}"
                 for a in core.PROTECTED]
        print(f"  {model:<18} " + "  ".join(parts))
    print(f"\n  (Organization usage  age:{core.usage(space.beta_org, cols, 'age_years'):.2f}  "
          f"sex:{core.usage(space.beta_org, cols, 'personal_status_sex'):.2f}  "
          f"foreign:{core.usage(space.beta_org, cols, 'foreign_worker'):.2f})")

    print(f"\nScatter fit: r = {r_all:+.3f} (p={p_all:.3f}); {r_ex:+.3f} excluding GPT-5.4-nano")
    print(f"\nSaved:\n  {scatter_png}\n  {usage_png}\n  {usage_csv}")


if __name__ == "__main__":
    main()
