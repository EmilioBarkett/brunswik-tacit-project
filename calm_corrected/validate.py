"""
calm_corrected.validate
=======================

Regression guard for the corrected Study 2 analysis. Runs ``run_alignment`` and
asserts the published anchors (see README "Validation values"):

  1. the 14 legacy ``repro_cos`` values (== the committed balanced-report cosines
     and the paper's published numbers) to 3 decimals;
  2. the legacy cosine-vs-accuracy Pearson r (the paper's German Credit r=0.15);
  3. the corrected-cosine spread (range, SD, count of cells > +0.15);
  4. the corrected cosine-vs-accuracy correlations (leverage-driven; NOT a
     headline finding — kept only as a precise regression anchor).

Exits nonzero with a clear per-check diff on any mismatch beyond tolerance.

TOLERANCE NOTE (do not "fix" by editing expected values)
--------------------------------------------------------
Twelve of the 14 ``repro_cos`` cells reproduce the committed reports exactly to
3 decimals. Two boundary cells — Grok 4.1 Fast [org-ext] and DeepSeek v3.2
[introspective] — compute to +0.177 / -0.184 on a numpy>=2 / current-BLAS stack
versus the committed +0.176 / -0.183, a sub-0.0015 solver/BLAS drift (verified
stable across scikit-learn 1.5-1.9; numpy 1.x does not build on Python 3.13).
This is numerical drift, NOT a method regression. ``REPRO_TOL`` absorbs it while
still failing hard on any real regression (the original NaN bug, a sign flip, or
the wrong construct all differ by >= 0.05). Expected values are left unchanged.
"""

import sys

import numpy as np
from scipy.stats import pearsonr

from calm_corrected import core
from calm_corrected import run_alignment

# Tolerances (documented; see module docstring).
REPRO_TOL = 0.0015   # absorbs numpy>=2 BLAS drift on 2 boundary cells
CORR_TOL = 0.005     # correlations / means
SD_TOL = 0.005

# Expected anchors (== section "Validation values" in the README == committed
# balanced reports == CELLS[*].paper_cos). Edit nothing here to chase a mismatch.
EXPECTED_REPRO = {(c.model, c.condition): c.paper_cos for c in core.CELLS}
EXPECTED_REPRO_PEARSON = 0.153
EXPECTED_CORR_RANGE = (-0.246, 0.207)
EXPECTED_CORR_SD = 0.117
EXPECTED_CORR_GT015 = 1
EXPECTED_CORRECTED_PEARSON = {"all": 0.769, "excl_nano": 0.365, "baseline": 0.951}
EXPECTED_CORRECTED_MEAN = {"baseline": -0.048, "org-ext": 0.081, "introspective": -0.027}


class Checker:
    def __init__(self):
        self.failures = []
        self.drifts = []

    def check(self, name, got, expected, tol, *, drift_tol=0.0):
        diff = abs(got - expected)
        if diff <= drift_tol and drift_tol > 0 and round(got, 3) != round(expected, 3):
            status, note = "DRIFT", f"  (numerical drift {diff:.4f} <= {drift_tol})"
            self.drifts.append(name)
        elif diff <= tol or round(got, 3) == round(expected, 3):
            status, note = "OK", ""
        else:
            status, note = "FAIL", f"  <-- diff {diff:+.4f} exceeds tol {tol}"
            self.failures.append(name)
        print(f"  [{status:4}] {name:<46} got={got:+.3f}  expected={expected:+.3f}{note}")


def main(argv=None):
    print("Running calm_corrected.run_alignment ...\n")
    df = run_alignment.main(argv if argv is not None else [])

    ck = Checker()

    print("\n=== 1. Legacy repro_cos by cell (3 decimals) ===")
    for cell in core.CELLS:
        got = float(df[(df.model == cell.model) & (df.condition == cell.condition)]["repro_cos"].iloc[0])
        exp = EXPECTED_REPRO[(cell.model, cell.condition)]
        ck.check(f"{cell.model} [{cell.condition}]", got, exp, tol=0.0, drift_tol=REPRO_TOL)

    print("\n=== 2. Legacy cosine-vs-accuracy Pearson r (paper's r=0.15) ===")
    r_repro = pearsonr(df["repro_cos"], df["acc"])[0]
    ck.check("repro_cos ~ accuracy (all cells)", r_repro, EXPECTED_REPRO_PEARSON, CORR_TOL)

    print("\n=== 3. Corrected-cosine spread (the robust signal) ===")
    c = df["corrected_cos"].dropna()
    ck.check("corrected range min", float(c.min()), EXPECTED_CORR_RANGE[0], CORR_TOL)
    ck.check("corrected range max", float(c.max()), EXPECTED_CORR_RANGE[1], CORR_TOL)
    ck.check("corrected SD", float(c.std()), EXPECTED_CORR_SD, SD_TOL)
    ck.check("corrected cells > +0.15", float((c > 0.15).sum()), float(EXPECTED_CORR_GT015), 0.0)
    for cond, exp in EXPECTED_CORRECTED_MEAN.items():
        got = float(df[df.condition == cond]["corrected_cos"].mean())
        ck.check(f"corrected mean [{cond}]", got, exp, CORR_TOL)

    print("\n=== 4. Corrected cosine-vs-accuracy correlation (leverage-driven; not a finding) ===")
    r_all = pearsonr(df["corrected_cos"], df["acc"])[0]
    sub = df[df.model != "GPT-5.4-nano"]
    r_ex = pearsonr(sub["corrected_cos"], sub["acc"])[0]
    base = df[df.condition == "baseline"]
    r_base = pearsonr(base["corrected_cos"], base["acc"])[0]
    ck.check("corrected_cos ~ accuracy (all cells)", r_all, EXPECTED_CORRECTED_PEARSON["all"], CORR_TOL)
    ck.check("corrected_cos ~ accuracy (excl. GPT-5.4-nano)", r_ex, EXPECTED_CORRECTED_PEARSON["excl_nano"], CORR_TOL)
    ck.check("corrected_cos ~ accuracy (baseline only)", r_base, EXPECTED_CORRECTED_PEARSON["baseline"], CORR_TOL)

    print()
    if ck.drifts:
        print(f"NOTE: {len(ck.drifts)} cell(s) within numerical-drift tolerance "
              f"({REPRO_TOL}): {', '.join(ck.drifts)}")
    if ck.failures:
        print(f"VALIDATION FAILED: {len(ck.failures)} mismatch(es) beyond tolerance:")
        for f in ck.failures:
            print(f"  - {f}")
        return 1
    print("VALIDATION PASSED: all anchors reproduce (within documented tolerance).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
