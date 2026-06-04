"""
calm_corrected.run_alignment
============================

Reproduces the paper's tier-label cosine (``repro_cos``) AND the corrected
cue-value cosine (``corrected_cos``) for all 14 cells (5 models x 3 conditions,
Grok introspective excluded), prints a table, reports cosine-vs-accuracy
correlations with robustness checks, and writes a results CSV.

HEADLINE (robust finding)
-------------------------
Corrected alignment is uniformly near zero (range ~[-0.25, +0.21]; only 1 of 14
cells exceeds +0.15). The corrected cosine-vs-accuracy correlation is NOT a
headline finding: it is leverage-driven (r = +0.77 over all cells but only +0.37
excluding GPT-5.4-nano; n=14, narrow range).

Usage:
    python -m calm_corrected.run_alignment --repo . --out-dir analysis/corrected
"""

import argparse
import os

from scipy.stats import pearsonr, spearmanr

from calm_corrected import core


def _corr_line(label, sub, col):
    s = sub.dropna(subset=[col])
    pr, pp = pearsonr(s[col], s["acc"])
    sr, sp = spearmanr(s[col], s["acc"])
    print(f"  {label:<24} n={len(s):>2}  Pearson={pr:+.3f} (p={pp:.3f})  "
          f"Spearman={sr:+.3f} (p={sp:.3f})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".", help="Path to repo root")
    ap.add_argument("--out-dir", default="analysis/corrected",
                    help="Directory for the results CSV (relative to --repo)")
    args = ap.parse_args(argv)

    out_dir = os.path.join(args.repo, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "corrected_german_credit_results.csv")

    df = core.compute_alignment_rows(args.repo)

    import pandas as pd
    pd.set_option("display.width", 140)
    print(df.to_string(index=False))
    df.to_csv(out_csv, index=False)

    # ----- cosine vs accuracy, legacy reproduction vs corrected -----
    print("\nCosine-vs-accuracy correlation:")
    print(" repro_cos (reproduces the paper's German Credit r=0.15):")
    _corr_line("all cells", df, "repro_cos")
    _corr_line("excl. GPT-5.4-nano", df[df.model != "GPT-5.4-nano"], "repro_cos")
    print(" corrected_cos:")
    _corr_line("all cells", df, "corrected_cos")
    _corr_line("excl. GPT-5.4-nano", df[df.model != "GPT-5.4-nano"], "corrected_cos")
    _corr_line("baseline only", df[df.condition == "baseline"], "corrected_cos")

    # ----- robust signal: the LEVEL of corrected alignment -----
    print("\nLevel of corrected alignment (the robust signal):")
    c = df["corrected_cos"].dropna()
    print(f"  range=[{c.min():+.3f}, {c.max():+.3f}]  SD={c.std():.3f}  "
          f"cells > +0.15: {int((c > 0.15).sum())}/{len(c)}")
    for cond, g in df.groupby("condition", sort=False):
        cc = g["corrected_cos"].dropna()
        print(f"  {cond:<13} mean={cc.mean():+.3f}  "
              f"range=[{cc.min():+.3f}, {cc.max():+.3f}]  n={len(cc)}")

    print(f"\nSaved: {out_csv}")
    return df


if __name__ == "__main__":
    main()
