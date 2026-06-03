"""
calm_corrected.core
===================

Shared logic for the *corrected* German Credit (Study 2) process-alignment
analysis. Both entry points import everything from here; there are no
copy-pasted duplicates.

WHY THIS PACKAGE EXISTS
-----------------------
The German Credit cosines reported in the paper come from ``analyze_weights.py``.
Two problems motivated this corrected reimplementation:

  Problem A — wrong construct. ``analyze_weights.py::encode_features`` builds the
    regression design matrix from ``merged[ALL_ATTRIBUTES]``, which after the
    merge are the LLM's own per-case tier labels (HIGH/MEDIUM/LOW), label
    encoded — NOT the actual case features (those sit unused in the ``*_dec``
    columns). The "organization" vector is the ground-truth outcome regressed on
    those same LLM tier labels, recomputed per model, so there is no fixed
    organizational policy. That is not the method the paper describes (decisions
    regressed on cue values against a fixed organizational policy).

  Problem B — non-reproducibility. Under current pandas the tier labels load as
    a string dtype, so ``encode_features``'s ``dtype == object`` test fails,
    routes them through ``pd.to_numeric().fillna(0)``, zeroes the design matrix,
    and the cosine becomes NaN. The committed reports in ``analysis/`` therefore
    no longer regenerate from HEAD. (Fixed in ``analyze_weights.py`` so the
    legacy path reproduces again; see ``LEGACY`` notes below.)

THE CORRECTED METHOD
--------------------
Each LLM's decisions, and the organization's historical decisions (the
``credit_risk`` labels), are regressed on the ACTUAL cue values — one-hot
encoded and standardized — in a SHARED feature space, against ONE fixed
organizational policy computed once on the balanced 600 cases. In the German
Credit data the target ``credit_risk`` IS the bank's historical Good/Bad
decision, so the organizational cue-utilization policy is a logistic regression
of ``credit_risk`` on the cue values.

CONSTRUCT CHOICES (flagged for co-author review; NOT yet co-author-validated)
----------------------------------------------------------------------------
These are surfaced as named constants below, not buried in the code:

  - RIDGE_C = 1.0            ridge strength, matching the original.
  - ORG_POLICY_SCOPE         the fixed org policy is fit ONCE on the balanced 600
                             cases (all Bad + a random_state=42 sample of Good).
  - ONE_HOT_DROP_FIRST=False coefficients are compared in the one-hot expanded
                             space (not aggregated back to 20 attributes).
  - USAGE metric             per-attribute "usage" = L2 norm of the
                             unit-normalized coefficients over that attribute's
                             one-hot columns.
  - BALANCE_SEED = 42        balanced subset replicated exactly from
                             analyze_weights.py.

LEGACY (paper-reproduction) PATH
--------------------------------
``legacy_cosine`` faithfully reproduces the originally reported tier-label
cosines (Problem A's construct), by label-encoding the LLM tier labels exactly
as ``analyze_weights.py`` does after its Problem-B fix. It is kept ONLY to
reproduce the published numbers; the corrected metric supersedes it.
"""

import json
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── CONSTRUCT CHOICES (flagged; change here, nowhere else) ───────────────────
RIDGE_C = 1.0                 # ridge strength for every logistic fit
BALANCE_SEED = 42             # random_state for the balanced Good subsample
RANDOM_STATE = 42             # LogisticRegression random_state
ONE_HOT_DROP_FIRST = False    # compare coefficients in the full one-hot space
CORRECTED_MAX_ITER = 2000     # solver iterations for the corrected path
LEGACY_MAX_ITER = 1000        # matches analyze_weights.py exactly (paper repro)

# ── CUE DEFINITIONS ──────────────────────────────────────────────────────────
ALL_ATTRIBUTES = [
    "checking_account_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account_bonds", "employment_since",
    "installment_rate_pct", "personal_status_sex", "other_debtors_guarantors",
    "present_residence_since", "property", "age_years", "other_installment_plans",
    "housing", "num_existing_credits", "job", "num_dependents", "telephone",
    "foreign_worker",
]
NUMERIC = [
    "duration_months", "credit_amount", "installment_rate_pct",
    "present_residence_since", "age_years", "num_existing_credits",
    "num_dependents",
]
CATEGORICAL = [a for a in ALL_ATTRIBUTES if a not in NUMERIC]

# Protected attributes + legitimate-cue anchors for the usage figure.
PROTECTED = ["age_years", "personal_status_sex", "foreign_worker"]
ANCHORS = ["checking_account_status", "duration_months", "credit_amount"]

# ── CELL -> FILE MAP (verified against committed balanced reports) ───────────
# This mapping is load-bearing: several court_extern_*/scenario_gpt_* files exist
# and only these exact files reproduce the committed numbers. Do not change it.
# ``load_results`` additionally asserts each file's model/condition columns match
# the expected cell and fails loudly otherwise.
Cell = namedtuple("Cell", ["model", "condition", "paper_cos", "path"])

CELLS = [
    Cell("Claude Haiku 4.5", "baseline",       0.503, "results/main/v2/results_german_credit_narratives_scenario_claude_20260415_073038.csv"),
    Cell("GPT-5.4-mini",     "baseline",       0.060, "results/main/v2/results_german_credit_narratives_scenario_gpt_20260416_094801.csv"),
    Cell("GPT-5.4-nano",     "baseline",       0.499, "results/main/v2/results_german_credit_narratives_scenario_gpt_20260416_094941.csv"),
    Cell("Grok 4.1 Fast",    "baseline",      -0.229, "results/main/v2/results_german_credit_narratives_scenario_ai_20260416_095006.csv"),
    Cell("DeepSeek v3.2",    "baseline",       0.264, "results/main/v2/results_german_credit_narratives_scenario_deepseek_20260416_095217.csv"),
    Cell("Claude Haiku 4.5", "org-ext",       -0.104, "results/main/v2/results_german_credit_narratives_court_extern_claude_20260416_124038.csv"),
    Cell("GPT-5.4-mini",     "org-ext",        0.224, "results/main/v2/results_german_credit_narratives_court_extern_gpt_20260416_124257.csv"),
    Cell("GPT-5.4-nano",     "org-ext",        0.076, "results/main/v2/results_german_credit_narratives_court_extern_gpt_20260416_124310.csv"),
    Cell("Grok 4.1 Fast",    "org-ext",        0.176, "results/main/v2/results_german_credit_narratives_court_extern_ai_20260416_124321.csv"),
    Cell("DeepSeek v3.2",    "org-ext",        0.164, "results/main/v2/results_german_credit_narratives_court_extern_deepseek_20260416_124332.csv"),
    Cell("Claude Haiku 4.5", "introspective",  0.360, "results/main/v2/results_german_credit_narratives_introspective_haiku_claude_20260427_142555.csv"),
    Cell("GPT-5.4-mini",     "introspective",  0.219, "results/main/v2/results_german_credit_narratives_introspective_gpt_mini_gpt_20260427_142606.csv"),
    Cell("GPT-5.4-nano",     "introspective", -0.256, "results/main/v2/results_german_credit_narratives_introspective_gpt_nano_gpt_20260427_142618.csv"),
    Cell("DeepSeek v3.2",    "introspective", -0.183, "results/main/v2/results_german_credit_narratives_introspective_deepseek_deepseek_20260427_142640.csv"),
    # Grok introspective is excluded in the paper (degenerate 99.5% Good over-correction).
]

# The single fixed organizational policy is fit on this cell's balanced 600.
ORG_POLICY_CELL = CELLS[0]  # Claude Haiku 4.5, baseline (scenario_claude_...)

# Expected raw values of the results CSV's `model`/`condition` columns, used to
# assert the cell->file mapping has not silently drifted.
MODEL_RAW = {
    "Claude Haiku 4.5": "anthropic/claude-haiku-4-5",
    "GPT-5.4-mini":     "openai/gpt-5.4-mini",
    "GPT-5.4-nano":     "openai/gpt-5.4-nano",
    "Grok 4.1 Fast":    "x-ai/grok-4.1-fast",
    "DeepSeek v3.2":    "deepseek/deepseek-v3.2",
}
CONDITION_RAW = {
    "baseline":      "scenario",
    "org-ext":       "court_extern",
    "introspective": "introspective",
}

# Display metadata for the figures.
CONDITION_DISPLAY = {
    "baseline":      ("baseline", "o"),
    "org-ext":       ("org-externalized", "s"),
    "introspective": ("introspective-externalized", "^"),
}
COLORS = {
    "Claude Haiku 4.5": "#1f77b4",
    "DeepSeek v3.2":    "#ff7f0e",
    "GPT-5.4-mini":     "#d62728",
    "GPT-5.4-nano":     "#9467bd",
    "Grok 4.1 Fast":    "#e377c2",
}


# ── PATH HELPERS ─────────────────────────────────────────────────────────────
def resolve(repo, rel):
    """Join a repo-relative path to the repo root (no hardcoded absolutes)."""
    import os
    return os.path.join(repo, rel)


def load_decoded(repo):
    """Load the decoded German Credit cue values (1-indexed to case_id later)."""
    return pd.read_csv(resolve(repo, "data/german_credit_decoded.csv"))


# ── LOADING / BALANCING ─────────────────────────────────────────────────────
def load_results(repo, cell):
    """Load a results CSV for ``cell``, attach parsed cue_weights, and assert
    that the file's model/condition columns match the expected cell."""
    df = pd.read_csv(resolve(repo, cell.path))

    exp_model = MODEL_RAW[cell.model]
    exp_cond = CONDITION_RAW[cell.condition]
    got_model = df["model"].iloc[0] if "model" in df.columns else None
    got_cond = df["condition"].iloc[0] if "condition" in df.columns else None
    if got_model != exp_model or got_cond != exp_cond:
        raise AssertionError(
            f"cell->file mapping mismatch for {cell.model} [{cell.condition}]\n"
            f"  file    : {cell.path}\n"
            f"  expected: model={exp_model!r} condition={exp_cond!r}\n"
            f"  found   : model={got_model!r} condition={got_cond!r}"
        )

    recs = []
    for _, row in df.iterrows():
        try:
            w = json.loads(row["cue_weights_json"])
            recs.append({a: None for a in ALL_ATTRIBUTES} if "error" in w else w)
        except Exception:
            recs.append({a: None for a in ALL_ATTRIBUTES})
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)


def balanced(res):
    """Replicates analyze_weights.py: all Bad + random_state=42 sample of Good,
    applied to the full loaded frame before any valid-classification filter."""
    bad = res[res["credit_risk"] == "Bad"]
    good = res[res["credit_risk"] == "Good"].sample(n=len(bad), random_state=BALANCE_SEED)
    return pd.concat([bad, good]).reset_index(drop=True)


def attach_values(frame, decoded):
    """Merge the actual decoded cue values onto a results frame by case_id."""
    dec = decoded.copy()
    dec.index = range(1, len(dec) + 1)
    dec.index.name = "case_id"
    dec = dec.reset_index()
    return frame.merge(dec, on="case_id", how="inner", suffixes=("", "_dec"))


def prepare_cell(repo, cell, decoded):
    """Balanced subset, restricted to valid Good/Bad classifications, with the
    binary LLM decision and the actual cue values attached. Used by BOTH the
    legacy and corrected paths so they operate on identical rows."""
    res = balanced(load_results(repo, cell))
    valid = res[res["classification"].isin(["Good", "Bad"])].copy()
    valid["llm_binary"] = (valid["classification"] == "Good").astype(int)
    return attach_values(valid, decoded)


def accuracy_good(merged):
    """Output accuracy (%) and Good-rate (%) for a prepared cell."""
    acc = (merged["classification"] == merged["credit_risk"]).mean() * 100
    good = merged["llm_binary"].mean() * 100
    return acc, good


# ── ENCODERS ────────────────────────────────────────────────────────────────
def encode_legacy(merged):
    """LEGACY (paper-reproduction) design matrix: the LLM's own tier labels,
    label-encoded. Faithful to analyze_weights.py after its Problem-B fix.
    Kept only to reproduce the originally reported cosines."""
    X = merged[ALL_ATTRIBUTES].copy()
    for c in ALL_ATTRIBUTES:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X.values


def build_value_frame(merged):
    """Actual cue values from the decoded ``*_dec`` columns, typed for one-hot
    + scaling (numerics coerced, categoricals as strings)."""
    cols = {a: merged[a + "_dec"] for a in ALL_ATTRIBUTES}
    X = pd.DataFrame(cols)
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    return X


# ── FIXED FEATURE SPACE + FIXED ORG POLICY ──────────────────────────────────
class FixedSpace:
    """The one-hot feature space and StandardScaler defined once on the org
    policy cell's balanced 600, plus the fixed organizational coefficient
    vector (``credit_risk`` regressed on the cue values)."""

    def __init__(self, columns, scaler, beta_org):
        self.columns = columns
        self.scaler = scaler
        self.beta_org = beta_org

    def design(self, value_frame):
        """One-hot, align to the fixed columns, then standardize."""
        d = pd.get_dummies(value_frame, columns=CATEGORICAL, drop_first=ONE_HOT_DROP_FIRST)
        d = d.reindex(columns=self.columns, fill_value=0)
        return self.scaler.transform(d.astype(float).values)


def build_fixed_space(repo, decoded):
    """Build the fixed feature space + fixed organizational policy ONCE, from
    the balanced 600 of ORG_POLICY_CELL (no valid-classification filter — the
    org policy is over the bank's historical decisions on all balanced cases)."""
    base = attach_values(balanced(load_results(repo, ORG_POLICY_CELL)), decoded)
    org_vals = build_value_frame(base)
    org_dummies = pd.get_dummies(org_vals, columns=CATEGORICAL, drop_first=ONE_HOT_DROP_FIRST)
    columns = org_dummies.columns
    scaler = StandardScaler().fit(org_dummies.astype(float).values)
    y_org = (base["credit_risk"] == "Good").astype(int).values
    beta_org = fit_coef(scaler.transform(org_dummies.astype(float).values), y_org, CORRECTED_MAX_ITER)
    return FixedSpace(columns, scaler, beta_org)


# ── REGRESSION / METRICS ─────────────────────────────────────────────────────
def fit_coef(X, y, max_iter):
    """Ridge-regularized logistic regression coefficient vector."""
    return LogisticRegression(max_iter=max_iter, random_state=RANDOM_STATE, C=RIDGE_C).fit(X, y).coef_[0]


def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else float("nan")


def legacy_cosine(merged):
    """Reproduce the paper's tier-label cosine for a prepared cell."""
    X = encode_legacy(merged)
    y_llm = merged["llm_binary"].values
    y_tru = (merged["credit_risk"] == "Good").astype(int).values
    return cosine(fit_coef(X, y_tru, LEGACY_MAX_ITER), fit_coef(X, y_llm, LEGACY_MAX_ITER))


def corrected_beta(merged, space):
    """The LLM's corrected cue-value coefficient vector in the fixed space."""
    X = space.design(build_value_frame(merged))
    return fit_coef(X, merged["llm_binary"].values, CORRECTED_MAX_ITER)


def corrected_cosine(merged, space):
    """Cosine between the LLM's corrected coefficients and the fixed org policy."""
    return cosine(space.beta_org, corrected_beta(merged, space))


# ── PER-ATTRIBUTE USAGE ──────────────────────────────────────────────────────
def cols_of(columns, attr):
    """Indices of the one-hot columns belonging to ``attr`` (exact match for a
    numeric column, ``attr_<value>`` for categorical dummies)."""
    return [i for i, c in enumerate(columns) if c == attr or c.startswith(attr + "_")]


def usage(beta, columns, attr):
    """Relative policy usage of ``attr``: L2 norm of the unit-normalized
    coefficient vector restricted to that attribute's columns."""
    b = beta / np.linalg.norm(beta)
    return float(np.linalg.norm(b[cols_of(columns, attr)]))


# ── HIGH-LEVEL DRIVERS ───────────────────────────────────────────────────────
def compute_alignment_rows(repo, decoded=None, space=None):
    """Return a DataFrame with one row per cell: n, acc, good, paper_cos,
    repro_cos (legacy reproduction) and corrected_cos."""
    if decoded is None:
        decoded = load_decoded(repo)
    if space is None:
        space = build_fixed_space(repo, decoded)

    rows = []
    for cell in CELLS:
        merged = prepare_cell(repo, cell, decoded)
        acc, good = accuracy_good(merged)
        rows.append(dict(
            model=cell.model, condition=cell.condition, n=len(merged),
            acc=round(acc, 1), good=round(good, 1),
            paper_cos=cell.paper_cos,
            repro_cos=round(legacy_cosine(merged), 3),
            corrected_cos=round(corrected_cosine(merged, space), 3),
        ))
    return pd.DataFrame(rows)


def cells_for(condition):
    """The cells (model -> Cell) for a given condition."""
    return {c.model: c for c in CELLS if c.condition == condition}
