"""
analyze_weights.py

Compares normative cue weights (from logistic regression on ground truth)
against LLM-implied cue weights, using two complementary methods:

PART 1 — DESCRIPTIVE
  Aggregates the tier labels (HIGH/MEDIUM/LOW/DISCOUNTED/NOT_MENTIONED)
  extracted by the scoring call across all cases. Compares directly to
  normative tiers. No regression required. Works on small samples.

PART 2 — REGRESSION-BASED
  Runs a logistic regression on the LLM's binary classifications using the
  original 20 cue values as predictors. Compares LLM-implied coefficients
  to normative coefficients. Requires the original decoded CSV and the
  results CSV. Needs 100+ cases to be meaningful.

Usage:
    # Descriptive only (works on any sample size)
    python analyze_weights.py --results results_bare_haiku.csv

    # Descriptive + regression (needs decoded CSV)
    python analyze_weights.py --results results_bare_haiku.csv --decoded german_credit_decoded.csv

    # Compare multiple models side by side
    python analyze_weights.py --results results_bare_haiku.csv results_bare_gpt4o.csv --decoded german_credit_decoded.csv

Arguments:
    --results   Path(s) to results CSV(s) from run_eval.py (required, one or more)
    --decoded   Path to german_credit_decoded.csv (required for regression)
    --normative Path to normative_cue_weights.csv (default: normative_cue_weights.csv)
    --output    Output directory for reports (default: analysis_output/)
"""

import os
import csv
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

ALL_ATTRIBUTES = [
    "checking_account_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account_bonds",
    "employment_since",
    "installment_rate_pct",
    "personal_status_sex",
    "other_debtors_guarantors",
    "present_residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "num_existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
]

TIER_ORDER = ["HIGH", "MEDIUM", "LOW", "DISCOUNTED", "NOT_MENTIONED"]

# Map tiers to numeric scores for correlation analysis
TIER_SCORES = {
    "HIGH": 4,
    "MEDIUM": 3,
    "LOW": 2,
    "DISCOUNTED": 1,
    "NOT_MENTIONED": 0,
}

# ── LOADERS ───────────────────────────────────────────────────────────────────

def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Parse cue_weights_json into separate columns
    weights_records = []
    for _, row in df.iterrows():
        try:
            w = json.loads(row["cue_weights_json"])
            if "error" in w:
                weights_records.append({a: None for a in ALL_ATTRIBUTES})
            else:
                weights_records.append(w)
        except Exception:
            weights_records.append({a: None for a in ALL_ATTRIBUTES})

    weights_df = pd.DataFrame(weights_records)
    return pd.concat([df.reset_index(drop=True), weights_df], axis=1)


def load_normative(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.set_index("original_attribute")
    return df


def load_decoded(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ── PART 1: DESCRIPTIVE ───────────────────────────────────────────────────────

def descriptive_analysis(df: pd.DataFrame, normative: pd.DataFrame, model_label: str) -> dict:
    """
    For each attribute, compute:
    - Most common LLM tier across all cases
    - % of cases where it was rated HIGH
    - % of cases where it was NOT_MENTIONED
    - Normative tier for comparison
    - Match flag
    """
    results = []

    for attr in ALL_ATTRIBUTES:
        col = df[attr].dropna()
        if len(col) == 0:
            continue

        counts = Counter(col)
        total  = len(col)

        most_common  = counts.most_common(1)[0][0]
        pct_high     = counts.get("HIGH", 0) / total * 100
        pct_medium   = counts.get("MEDIUM", 0) / total * 100
        pct_low      = counts.get("LOW", 0) / total * 100
        pct_disc     = counts.get("DISCOUNTED", 0) / total * 100
        pct_none     = counts.get("NOT_MENTIONED", 0) / total * 100

        norm_tier    = normative.loc[attr, "weight_tier"] if attr in normative.index else "UNKNOWN"
        norm_rank    = int(normative.loc[attr, "rank"]) if attr in normative.index else 99
        tier_match   = most_common == norm_tier

        # Mean numeric score
        numeric_scores = [TIER_SCORES[t] for t in col if t in TIER_SCORES]
        mean_score = np.mean(numeric_scores) if numeric_scores else 0

        results.append({
            "attribute":       attr,
            "norm_rank":       norm_rank,
            "norm_tier":       norm_tier,
            "llm_most_common": most_common,
            "llm_mean_score":  round(mean_score, 2),
            "pct_HIGH":        round(pct_high, 1),
            "pct_MEDIUM":      round(pct_medium, 1),
            "pct_LOW":         round(pct_low, 1),
            "pct_DISCOUNTED":  round(pct_disc, 1),
            "pct_NOT_MENTIONED": round(pct_none, 1),
            "tier_match":      tier_match,
            "n_cases":         total,
        })

    result_df = pd.DataFrame(results).sort_values("norm_rank")

    # Spearman correlation between normative rank and LLM mean score rank
    llm_ranks = result_df["llm_mean_score"].rank(ascending=False)
    rho, pval = spearmanr(result_df["norm_rank"], llm_ranks)

    return {
        "df":          result_df,
        "rho":         rho,
        "pval":        pval,
        "match_count": result_df["tier_match"].sum(),
        "total":       len(result_df),
        "model":       model_label,
        "n_cases":     len(df),
    }


# ── PART 2: REGRESSION-BASED ──────────────────────────────────────────────────

def regression_analysis(
    results_df: pd.DataFrame,
    decoded_df: pd.DataFrame,
    normative: pd.DataFrame,
    model_label: str,
) -> dict:
    """
    Runs logistic regression on the LLM's classifications using the original
    20 cue values as predictors. Compares implied coefficients to normative.

    Only uses cases where classification is Good or Bad (no API_ERROR/PARSE_FAILED).
    Merges results with decoded CSV on case_id (1-indexed).
    """
    # Filter to valid classifications only
    valid = results_df[results_df["classification"].isin(["Good", "Bad"])].copy()
    valid["llm_binary"] = (valid["classification"] == "Good").astype(int)

    if len(valid) < 20:
        return {
            "error": f"Only {len(valid)} valid cases — need at least 20 for regression. Run more cases first.",
            "model": model_label,
        }

    # Decoded CSV is 1-indexed to match case_id
    decoded_df = decoded_df.copy()
    decoded_df.index = range(1, len(decoded_df) + 1)
    decoded_df.index.name = "case_id"
    decoded_df = decoded_df.reset_index()

    # Merge on case_id
    merged = valid.merge(decoded_df, on="case_id", how="inner", suffixes=("", "_decoded"))

    if len(merged) < 20:
        return {
            "error": f"Only {len(merged)} cases after merge — check case_id alignment.",
            "model": model_label,
        }

    # Encode categorical columns
    X_cols = ALL_ATTRIBUTES
    X = merged[X_cols].copy()

    # Label-encode all columns (handles both categorical and numeric)
    encoders = {}
    for col in X_cols:
        if X[col].dtype == object:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    y = merged["llm_binary"].values

    # Fit logistic regression with L2 regularization
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X, y)

    # Extract absolute coefficients (magnitude = importance)
    coef_abs = np.abs(clf.coef_[0])

    # Build comparison dataframe
    coef_df = pd.DataFrame({
        "attribute":     X_cols,
        "llm_coef_abs":  coef_abs,
        "llm_coef_raw":  clf.coef_[0],
    })

    # Merge with normative
    norm_subset = normative[["normative_weight", "rank", "weight_tier"]].rename(
        columns={"normative_weight": "norm_weight", "rank": "norm_rank", "weight_tier": "norm_tier"}
    )
    coef_df = coef_df.merge(
        norm_subset.reset_index().rename(columns={"original_attribute": "attribute"}),
        on="attribute", how="left"
    )

    # LLM implied rank
    coef_df["llm_rank"] = coef_df["llm_coef_abs"].rank(ascending=False).astype(int)

    # Spearman correlation between normative rank and LLM implied rank
    rho, pval = spearmanr(coef_df["norm_rank"], coef_df["llm_rank"])

    coef_df = coef_df.sort_values("norm_rank")

    return {
        "df":         coef_df,
        "rho":        rho,
        "pval":       pval,
        "n_cases":    len(merged),
        "model":      model_label,
        "accuracy":   clf.score(X, y),
    }


# ── REPORT WRITER ─────────────────────────────────────────────────────────────

def write_report(desc: dict, reg: dict | None, output_path: str):
    model   = desc["model"]
    n_cases = desc["n_cases"]
    lines   = []

    lines.append("=" * 70)
    lines.append(f"CUE WEIGHT ANALYSIS REPORT")
    lines.append(f"Model:    {model}")
    lines.append(f"Cases:    {n_cases}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # ── PART 1: DESCRIPTIVE ───────────────────────────────────────────────
    lines.append("\n── PART 1: DESCRIPTIVE TIER COMPARISON ──────────────────────────────\n")
    lines.append(
        f"{'Attribute':<35} {'Norm':<8} {'LLM':<18} {'Match':<7} {'%HIGH':<7} {'%NONE'}"
    )
    lines.append("-" * 85)

    df = desc["df"]
    for _, row in df.iterrows():
        match_str = "✓" if row["tier_match"] else "✗"
        lines.append(
            f"{row['attribute']:<35} "
            f"{row['norm_tier']:<8} "
            f"{row['llm_most_common']:<18} "
            f"{match_str:<7} "
            f"{row['pct_HIGH']:<7.1f} "
            f"{row['pct_NOT_MENTIONED']:.1f}"
        )

    match_pct = desc["match_count"] / desc["total"] * 100
    lines.append(f"\nTier match rate: {desc['match_count']}/{desc['total']} ({match_pct:.1f}%)")
    lines.append(f"Spearman ρ (normative rank vs LLM weight rank): {desc['rho']:.3f}  p={desc['pval']:.3f}")

    # ── OVER/UNDER WEIGHTING SUMMARY ─────────────────────────────────────
    lines.append("\n── OVER-WEIGHTED CUES (LLM rates higher than normative) ────────────\n")
    for _, row in df.iterrows():
        llm_score  = TIER_SCORES.get(row["llm_most_common"], 0)
        norm_score = TIER_SCORES.get(row["norm_tier"], 0)
        if llm_score > norm_score:
            lines.append(
                f"  {row['attribute']:<35} norm={row['norm_tier']}  llm={row['llm_most_common']}  "
                f"(norm rank #{row['norm_rank']})"
            )

    lines.append("\n── UNDER-WEIGHTED CUES (LLM rates lower than normative) ────────────\n")
    for _, row in df.iterrows():
        llm_score  = TIER_SCORES.get(row["llm_most_common"], 0)
        norm_score = TIER_SCORES.get(row["norm_tier"], 0)
        if llm_score < norm_score:
            lines.append(
                f"  {row['attribute']:<35} norm={row['norm_tier']}  llm={row['llm_most_common']}  "
                f"(norm rank #{row['norm_rank']})"
            )

    # ── PART 2: REGRESSION ────────────────────────────────────────────────
    lines.append("\n── PART 2: REGRESSION-BASED COEFFICIENT COMPARISON ─────────────────\n")

    if reg is None:
        lines.append("  Skipped — no decoded CSV provided.")
    elif "error" in reg:
        lines.append(f"  Skipped — {reg['error']}")
    else:
        reg_df = reg["df"]
        lines.append(
            f"{'Attribute':<35} {'Norm rank':<12} {'LLM rank':<12} {'Norm weight':<14} {'LLM |coef|'}"
        )
        lines.append("-" * 85)
        for _, row in reg_df.iterrows():
            lines.append(
                f"{row['attribute']:<35} "
                f"#{int(row['norm_rank']):<11} "
                f"#{int(row['llm_rank']):<11} "
                f"{row['norm_weight']:<14.4f} "
                f"{row['llm_coef_abs']:.4f}"
            )

        lines.append(f"\nCases used for regression: {reg['n_cases']}")
        lines.append(f"LLM regression accuracy (in-sample): {reg['accuracy']*100:.1f}%")
        lines.append(
            f"Spearman ρ (normative rank vs LLM implied rank): {reg['rho']:.3f}  p={reg['pval']:.3f}"
        )
        lines.append(
            "\nInterpretation: ρ close to 1.0 = LLM weights cues similarly to normative model.\n"
            "ρ close to 0 or negative = LLM weights diverge substantially from ground truth."
        )

    lines.append("\n" + "=" * 70)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze LLM vs normative cue weights.")
    parser.add_argument("--results",   nargs="+", required=True, help="Path(s) to results CSV(s)")
    parser.add_argument("--decoded",   default=None,             help="Path to german_credit_decoded.csv")
    parser.add_argument("--normative", default="normative_cue_weights.csv", help="Path to normative weights CSV")
    parser.add_argument("--output",    default="analysis_output", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load normative weights
    if not os.path.exists(args.normative):
        raise FileNotFoundError(f"Normative weights not found: {args.normative}")
    normative = load_normative(args.normative)

    # Load decoded CSV if provided
    decoded_df = None
    if args.decoded:
        if not os.path.exists(args.decoded):
            print(f"Warning: decoded CSV not found at {args.decoded} — skipping regression.")
        else:
            decoded_df = load_decoded(args.decoded)

    # Run analysis for each results file
    all_desc = []
    for results_path in args.results:
        if not os.path.exists(results_path):
            print(f"Warning: results file not found: {results_path} — skipping.")
            continue

        print(f"\nAnalysing: {results_path}")
        results_df  = load_results(results_path)
        model_label = results_df["model"].iloc[0] if "model" in results_df.columns else results_path

        # Part 1: Descriptive
        desc = descriptive_analysis(results_df, normative, model_label)
        all_desc.append(desc)

        # Part 2: Regression
        reg = None
        if decoded_df is not None:
            reg = regression_analysis(results_df, decoded_df, normative, model_label)

        # Write individual report
        stem        = os.path.splitext(os.path.basename(results_path))[0]
        report_path = os.path.join(args.output, f"report_{stem}.txt")
        write_report(desc, reg, report_path)

        # Save descriptive CSV
        csv_path = os.path.join(args.output, f"descriptive_{stem}.csv")
        desc["df"].to_csv(csv_path, index=False)
        print(f"\nDescriptive CSV saved: {csv_path}")
        print(f"Report saved:          {report_path}")

    # ── MULTI-MODEL COMPARISON ────────────────────────────────────────────
    if len(all_desc) > 1:
        print("\n── MULTI-MODEL COMPARISON ───────────────────────────────────────────\n")
        comp_lines = []
        comp_lines.append(f"{'Attribute':<35} {'Norm':<8}", )
        for d in all_desc:
            label = d["model"].split("/")[-1][:12]
            comp_lines[0] = comp_lines[0] if isinstance(comp_lines[0], str) else ""

        # Rebuild header with all models
        header = f"{'Attribute':<35} {'Norm':<8}"
        for d in all_desc:
            label = d["model"].split("/")[-1][:14]
            header += f" {label:<16}"
        comp_lines = [header, "-" * (35 + 8 + 16 * len(all_desc) + 4)]

        for attr in ALL_ATTRIBUTES:
            norm_tier = normative.loc[attr, "weight_tier"] if attr in normative.index else "?"
            row_str   = f"{attr:<35} {norm_tier:<8}"
            for d in all_desc:
                attr_row = d["df"][d["df"]["attribute"] == attr]
                if len(attr_row) > 0:
                    llm_tier = attr_row.iloc[0]["llm_most_common"]
                    match    = "✓" if attr_row.iloc[0]["tier_match"] else "✗"
                    row_str += f" {llm_tier:<14}{match} "
                else:
                    row_str += f" {'N/A':<16}"
            comp_lines.append(row_str)

        comp_lines.append("\nSpearman ρ summary:")
        for d in all_desc:
            comp_lines.append(f"  {d['model']}: ρ={d['rho']:.3f}  p={d['pval']:.3f}  match={d['match_count']}/{d['total']}")

        comp_text  = "\n".join(comp_lines)
        comp_path  = os.path.join(args.output, "multi_model_comparison.txt")
        with open(comp_path, "w") as f:
            f.write(comp_text)
        print(comp_text)
        print(f"\nMulti-model comparison saved: {comp_path}")


if __name__ == "__main__":
    main()
