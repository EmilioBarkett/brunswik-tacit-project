"""
plot_credit_panel.py

Renders ONLY the German Credit (Study 2) panel of Figure 1, styled identically
to the right panel of plot_comparison_figure.py, so it can replace the old
subplot directly:

  - corrected cue-value cosines (calm_corrected), balanced-600 accuracies;
  - same per-model colours and per-condition markers;
  - green dotted "bank regression linear ceiling" line at 0.715 (same hue as the
    combined figure);
  - NO in-plot legend (the shared legend sits below the combined figure);
  - single Pearson r in the title, matching the ECHR panel.

Output:
  report/icml2026/figures/corrected_german_credit_panel.png

Usage:
  python plot_credit_panel.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot_comparison_figure import GERMAN_RECORDS, plot_panel

OUTPUT = "report/icml2026/figures/corrected_german_credit_panel.png"


def main():
    gc = pd.DataFrame(GERMAN_RECORDS, columns=["model", "condition", "cosine", "accuracy"])

    # One panel at the same per-panel size as the combined figure ((12, 5) / 2).
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plot_panel(
        ax, gc,
        x_col="cosine", y_col="accuracy",
        condition_col="condition", model_col="model",
        title="Study 2: German Credit Decisions\n(balanced subset, $n=5$ models × 2–3 conditions)",
        xlabel="Policy alignment: cosine similarity of ridge cue weights",
        ylabel="LLM accuracy vs. ground truth labels",
        normative_line=0.715, normative_label="bank regression linear ceiling",
    )
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {OUTPUT}")


if __name__ == "__main__":
    main()
