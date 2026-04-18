"""
analyze_weights.py

Compares normative cue weights (from logistic regression on ground truth)
against LLM-implied cue weights, using metrics that mirror those used in
the collaborator's ECHR analysis report.

PRIMARY METRIC  — Coefficient cosine similarity: angular similarity between
                  the normative and LLM signed coefficient vectors from
                  ridge-regularized logistic regression. 1.0 = identical
                  direction; 0 = orthogonal; negative = opposed.

SECONDARY METRICS
  Pearson r          : linear correlation between the same coefficient vectors
  Propensity corr.   : correlation between normative and LLM per-case
                       predicted probabilities (cross-validated)
  ROC AUC            : how predictable LLM decisions are from the 20 cues
                       (higher = more systematic/cue-driven)
  Output accuracy    : % of LLM classifications matching ground truth
  Cohen's kappa      : chance-corrected agreement with ground truth
  Good rate          : % of cases the LLM classified as Good
                       (dataset base rate = 70%; deviations = threshold bias)

DESCRIPTIVE TIER COMPARISON
  Aggregates tier labels (HIGH/MEDIUM/LOW/DISCOUNTED/NOT_MENTIONED)
  extracted per case by the scoring call. Works on small samples.
  Unique to this project — not in the ECHR report.

BOOTSTRAP SIGNIFICANCE
  When multiple results files are passed, computes Δ cosine similarity
  between conditions with bootstrap p-values (200 replicates).

Usage:
    # Single model / condition
    python analyze_weights.py \\
        --results results/main/results_scenario_haiku.csv \\
        --decoded data/german_credit_decoded.csv

    # Multiple conditions — triggers bootstrap comparison
    python analyze_weights.py \\
        --results results/main/results_scenario_haiku.csv \\
                  results/main/results_directive_haiku.csv \\
        --decoded data/german_credit_decoded.csv

Arguments:
    --results        Path(s) to results CSV(s) from run_eval.py (required)
    --decoded        Path to german_credit_decoded.csv (required for regression)
    --normative      Path to normative_cue_weights.csv
                     (default: data/normative_cue_weights.csv)
    --propensities   Path to normative_propensities.csv
                     (default: data/normative_propensities.csv)
    --output         Output directory (default: analysis/)
    --bootstrap-n    Number of bootstrap replicates (default: 200)
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from sklearn.model_selection import cross_val_predict
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings("ignore")

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

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

TIER_SCORES = {
    "HIGH": 4,
    "MEDIUM": 3,
    "LOW": 2,
    "DISCOUNTED": 1,
    "NOT_MENTIONED": 0,
}

NORMATIVE_AUC      = 0.751   # from normative_weights.py 5-fold CV accuracy
NORMATIVE_GOOD_PCT = 70.0    # dataset base rate

# ── LOADERS ────────────────────────────────────────────────────────────────────

def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
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
    return df.set_index("original_attribute")


def load_decoded(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_propensities(path: str) -> pd.DataFrame:
    """Normative per-case P(Good) from cross_val_predict in normative_weights.py."""
    return pd.read_csv(path)[["case_id", "normative_prob_good"]]


# ── FEATURE ENCODING ───────────────────────────────────────────────────────────

def encode_features(X_raw: pd.DataFrame) -> np.ndarray:
    """
    Label-encode categorical columns, coerce numerics.
    Returns a numpy array with shape (n_cases, 20).
    Encoding is fit on the passed data, so it is consistent within a call.
    """
    X = X_raw.copy()
    for col in ALL_ATTRIBUTES:
        if X[col].dtype == object:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    return X[ALL_ATTRIBUTES].values


# ── OUTPUT METRICS ─────────────────────────────────────────────────────────────

def output_metrics(results_df: pd.DataFrame, model_label: str) -> dict:
    """
    Accuracy, Cohen's kappa, and Good rate — computed directly from the
    classification column without needing the decoded CSV.
    """
    valid = results_df[results_df["classification"].isin(["Good", "Bad"])].copy()
    n     = len(valid)

    if n == 0:
        return {"model": model_label, "error": "No valid classifications."}

    correct   = (valid["classification"] == valid["credit_risk"]).sum()
    accuracy  = correct / n

    y_true = (valid["credit_risk"]    == "Good").astype(int).values
    y_pred = (valid["classification"] == "Good").astype(int).values

    kappa     = cohen_kappa_score(y_true, y_pred)
    good_rate = y_pred.mean() * 100

    parse_failed = (results_df["classification"] == "PARSE_FAILED").sum()
    api_errors   = (results_df["classification"] == "API_ERROR").sum()

    return {
        "model":        model_label,
        "n_valid":      n,
        "n_total":      len(results_df),
        "parse_failed": parse_failed,
        "api_errors":   api_errors,
        "accuracy":     accuracy,
        "kappa":        kappa,
        "good_rate":    good_rate,
    }


# ── DESCRIPTIVE TIER ANALYSIS ──────────────────────────────────────────────────

def descriptive_analysis(results_df: pd.DataFrame, normative: pd.DataFrame,
                         model_label: str) -> dict:
    """
    Aggregates scoring-model tier labels across all cases per attribute.
    Works on any sample size. Unique to this project.
    """
    rows = []
    for attr in ALL_ATTRIBUTES:
        col   = results_df[attr].dropna()
        if len(col) == 0:
            continue
        counts = Counter(col)
        total  = len(col)

        most_common = counts.most_common(1)[0][0]
        norm_tier   = normative.loc[attr, "weight_tier"] if attr in normative.index else "UNKNOWN"
        norm_rank   = int(normative.loc[attr, "rank"])   if attr in normative.index else 99

        numeric_scores = [TIER_SCORES[t] for t in col if t in TIER_SCORES]
        mean_score     = np.mean(numeric_scores) if numeric_scores else 0.0

        rows.append({
            "attribute":         attr,
            "norm_rank":         norm_rank,
            "norm_tier":         norm_tier,
            "llm_most_common":   most_common,
            "llm_mean_score":    round(mean_score, 2),
            "pct_HIGH":          round(counts.get("HIGH",          0) / total * 100, 1),
            "pct_MEDIUM":        round(counts.get("MEDIUM",        0) / total * 100, 1),
            "pct_LOW":           round(counts.get("LOW",           0) / total * 100, 1),
            "pct_DISCOUNTED":    round(counts.get("DISCOUNTED",    0) / total * 100, 1),
            "pct_NOT_MENTIONED": round(counts.get("NOT_MENTIONED", 0) / total * 100, 1),
            "tier_match":        most_common == norm_tier,
            "n_cases":           total,
        })

    result_df = pd.DataFrame(rows).sort_values("norm_rank")

    llm_ranks  = result_df["llm_mean_score"].rank(ascending=False)
    rho, rpval = spearmanr(result_df["norm_rank"], llm_ranks)

    return {
        "df":          result_df,
        "rho":         rho,
        "rho_pval":    rpval,
        "match_count": int(result_df["tier_match"].sum()),
        "total":       len(result_df),
        "model":       model_label,
        "n_cases":     len(results_df),
    }


# ── REGRESSION-BASED ANALYSIS ──────────────────────────────────────────────────

def regression_analysis(results_df: pd.DataFrame, decoded_df: pd.DataFrame,
                        normative: pd.DataFrame, propensities_df: pd.DataFrame | None,
                        model_label: str) -> dict:
    """
    Primary analysis. Fits ridge-regularized logistic regression on LLM
    classifications using the 20 original cue values as predictors.

    Metrics produced:
      cosine_sim    — angular similarity between normative and LLM signed coef vectors
      pearson_r     — linear correlation between the same vectors
      prop_corr     — propensity correlation (per-case P(Good) between models)
      auc           — how predictable LLM decisions are from cues (cross-validated)
      spearman_rho  — rank correlation between normative and LLM coefficient ranks
                      (kept as supplementary; Spearman on abs coefs)
    """
    valid = results_df[results_df["classification"].isin(["Good", "Bad"])].copy()
    valid["llm_binary"] = (valid["classification"] == "Good").astype(int)

    if len(valid) < 50:
        return {
            "error": f"Only {len(valid)} valid cases — need at least 50 for reliable regression.",
            "model": model_label,
        }

    # Align decoded CSV to 1-indexed case_ids
    decoded = decoded_df.copy()
    decoded.index = range(1, len(decoded) + 1)
    decoded.index.name = "case_id"
    decoded = decoded.reset_index()

    merged = valid.merge(decoded, on="case_id", how="inner", suffixes=("", "_dec"))

    if len(merged) < 50:
        return {
            "error": f"Only {len(merged)} cases after merge — check case_id alignment.",
            "model": model_label,
        }

    X       = encode_features(merged[ALL_ATTRIBUTES])
    y_llm   = merged["llm_binary"].values
    y_truth = (merged["credit_risk"] == "Good").astype(int).values

    # ── FIT LLM REGRESSION ────────────────────────────────────────────────
    clf_llm = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf_llm.fit(X, y_llm)
    llm_coef = clf_llm.coef_[0]   # signed, 20-dim

    # ── REFIT NORMATIVE REGRESSION ON SAME CASES & ENCODING ───────────────
    # Using the same feature space so the two coefficient vectors are
    # directly comparable (same 20 dimensions, same label encoding).
    clf_norm = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf_norm.fit(X, y_truth)
    norm_coef = clf_norm.coef_[0]  # signed, 20-dim

    # ── PRIMARY: COSINE SIMILARITY ────────────────────────────────────────
    cosine_sim = (
        np.dot(norm_coef, llm_coef)
        / (np.linalg.norm(norm_coef) * np.linalg.norm(llm_coef))
    )

    # ── SECONDARY: PEARSON r ──────────────────────────────────────────────
    pearson_r, pearson_p = pearsonr(norm_coef, llm_coef)

    # ── ROC AUC (cross-validated) ─────────────────────────────────────────
    # How well do the 20 cues predict the LLM's own decisions?
    llm_proba_cv = cross_val_predict(
        clf_llm, X, y_llm, cv=5, method="predict_proba"
    )
    auc = roc_auc_score(y_llm, llm_proba_cv[:, 1])

    # ── PROPENSITY CORRELATION ────────────────────────────────────────────
    prop_corr = None
    if propensities_df is not None:
        norm_props = propensities_df.merge(
            merged[["case_id"]], on="case_id", how="inner"
        )
        llm_proba_full = cross_val_predict(
            clf_llm, X, y_llm, cv=5, method="predict_proba"
        )
        llm_props = pd.DataFrame({
            "case_id":       merged["case_id"].values,
            "llm_prob_good": llm_proba_full[:, 1],
        })
        both = norm_props.merge(llm_props, on="case_id", how="inner")
        if len(both) >= 10:
            prop_corr, _ = pearsonr(
                both["normative_prob_good"], both["llm_prob_good"]
            )

    # ── SUPPLEMENTARY: SPEARMAN ρ ON ABS COEFFICIENT RANKS ───────────────
    coef_abs   = np.abs(llm_coef)
    norm_abs   = np.abs(norm_coef)
    llm_ranks  = pd.Series(coef_abs).rank(ascending=False).values
    norm_ranks = pd.Series(norm_abs).rank(ascending=False).values
    spearman_rho, spearman_p = spearmanr(norm_ranks, llm_ranks)

    # ── COEFFICIENT COMPARISON TABLE ──────────────────────────────────────
    coef_df = pd.DataFrame({
        "attribute":    ALL_ATTRIBUTES,
        "norm_coef":    norm_coef,
        "llm_coef":     llm_coef,
        "norm_coef_abs": norm_abs,
        "llm_coef_abs": coef_abs,
    })
    coef_df["norm_rank"] = coef_df["norm_coef_abs"].rank(ascending=False).astype(int)
    coef_df["llm_rank"]  = coef_df["llm_coef_abs"].rank(ascending=False).astype(int)

    # Add normative weight tier from the normative CSV for reference
    norm_tiers = normative["weight_tier"].rename("norm_tier")
    coef_df = coef_df.merge(
        norm_tiers.reset_index().rename(columns={"original_attribute": "attribute"}),
        on="attribute", how="left"
    )
    coef_df = coef_df.sort_values("norm_rank")

    return {
        "df":           coef_df,
        "cosine_sim":   cosine_sim,
        "pearson_r":    pearson_r,
        "pearson_p":    pearson_p,
        "prop_corr":    prop_corr,
        "auc":          auc,
        "spearman_rho": spearman_rho,
        "spearman_p":   spearman_p,
        "n_cases":      len(merged),
        "model":        model_label,
    }


# ── BOOTSTRAP Δ COSINE ────────────────────────────────────────────────────────

def bootstrap_delta_cosine(reg_a: dict, reg_b: dict, decoded_df: pd.DataFrame,
                           results_a: pd.DataFrame, results_b: pd.DataFrame,
                           n_boot: int = 200) -> dict:
    """
    Bootstrap significance test for Δ cosine similarity between two conditions.
    Resamples cases with replacement and recomputes cosine for each condition.
    p-value = proportion of bootstrap samples where Δ ≤ 0 (one-sided, positive Δ).
    """
    if "error" in reg_a or "error" in reg_b:
        return {"error": "Cannot bootstrap — regression failed for one or more conditions."}

    decoded = decoded_df.copy()
    decoded.index = range(1, len(decoded) + 1)
    decoded.index.name = "case_id"
    decoded = decoded.reset_index()

    def get_cosine(results_df: pd.DataFrame, case_ids: np.ndarray) -> float:
        valid = results_df[results_df["classification"].isin(["Good", "Bad"])].copy()
        valid["llm_binary"] = (valid["classification"] == "Good").astype(int)
        subset = valid[valid["case_id"].isin(case_ids)]
        merged = subset.merge(decoded, on="case_id", how="inner", suffixes=("", "_dec"))
        if len(merged) < 20:
            return np.nan
        X      = encode_features(merged[ALL_ATTRIBUTES])
        y_llm  = merged["llm_binary"].values
        y_true = (merged["credit_risk"] == "Good").astype(int).values
        clf_l  = LogisticRegression(max_iter=500, random_state=42, C=1.0).fit(X, y_llm)
        clf_n  = LogisticRegression(max_iter=500, random_state=42, C=1.0).fit(X, y_true)
        lc, nc = clf_l.coef_[0], clf_n.coef_[0]
        return np.dot(lc, nc) / (np.linalg.norm(lc) * np.linalg.norm(nc))

    # Shared case_ids across both conditions
    ids_a = set(results_a[results_a["classification"].isin(["Good","Bad"])]["case_id"])
    ids_b = set(results_b[results_b["classification"].isin(["Good","Bad"])]["case_id"])
    shared_ids = np.array(sorted(ids_a & ids_b))

    if len(shared_ids) < 50:
        return {"error": f"Only {len(shared_ids)} shared cases — need 50+ for bootstrap."}

    observed_delta = reg_b["cosine_sim"] - reg_a["cosine_sim"]
    deltas = []
    rng = np.random.default_rng(42)

    for _ in range(n_boot):
        sample = rng.choice(shared_ids, size=len(shared_ids), replace=True)
        cos_a  = get_cosine(results_a, sample)
        cos_b  = get_cosine(results_b, sample)
        if not (np.isnan(cos_a) or np.isnan(cos_b)):
            deltas.append(cos_b - cos_a)

    if not deltas:
        return {"error": "All bootstrap resamples failed."}

    deltas = np.array(deltas)
    if observed_delta >= 0:
        p_val = np.mean(deltas <= 0)
    else:
        p_val = np.mean(deltas >= 0)

    return {
        "observed_delta": observed_delta,
        "p_value":        p_val,
        "n_boot":         len(deltas),
        "significant":    p_val < 0.05,
        "ci_lower":       np.percentile(deltas, 2.5),
        "ci_upper":       np.percentile(deltas, 97.5),
    }


# ── REPORT WRITER ──────────────────────────────────────────────────────────────

def write_report(out: dict, output_path: str):
    """
    Writes a structured report for one model/condition combining all metrics.
    """
    model   = out["model"]
    n_cases = out["om"]["n_valid"]
    lines   = []

    W = 72

    lines.append("=" * W)
    lines.append("CUE UTILIZATION ALIGNMENT REPORT")
    lines.append(f"Model     : {model}")
    lines.append(f"Condition : {out.get('condition', 'unknown')}")
    lines.append(f"Cases     : {n_cases}  (total submitted: {out['om']['n_total']})")
    lines.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * W)

    # ── SECTION 1: SUMMARY METRICS ────────────────────────────────────────
    lines.append("\n── SECTION 1: SUMMARY METRICS ───────────────────────────────────────\n")
    lines.append(f"  {'Metric':<35} {'Value':<12} {'Normative benchmark'}")
    lines.append("  " + "-" * 62)

    om  = out["om"]
    reg = out.get("reg")

    lines.append(f"  {'Output accuracy':<35} {om['accuracy']*100:<12.1f}% {'75.1% (logistic regression)'}")
    lines.append(f"  {'Cohen\'s kappa':<35} {om['kappa']:<12.3f} {'1.0 = perfect agreement'}")
    lines.append(f"  {'Good classification rate':<35} {om['good_rate']:<12.1f}% {'70.0% (dataset base rate)'}")

    if reg and "error" not in reg:
        lines.append(f"  {'ROC AUC (cue-driven, CV)':<35} {reg['auc']:<12.3f} {'0.751 (normative model)'}")
        lines.append(f"  {'Cosine similarity [PRIMARY]':<35} {reg['cosine_sim']:<12.3f} {'1.0 = identical weighting'}")
        lines.append(f"  {'Pearson r (coefficients)':<35} {reg['pearson_r']:<12.3f} {'p={:.3f}'.format(reg['pearson_p'])}")
        if reg["prop_corr"] is not None:
            lines.append(f"  {'Propensity correlation':<35} {reg['prop_corr']:<12.3f} {'normative vs LLM P(Good)'}")
        lines.append(f"  {'Spearman ρ (rank, supplementary)':<35} {reg['spearman_rho']:<12.3f} {'p={:.3f}'.format(reg['spearman_p'])}")
    elif reg and "error" in reg:
        lines.append(f"\n  Regression skipped: {reg['error']}")

    if om["parse_failed"] or om["api_errors"]:
        lines.append(f"\n  Parse failures: {om['parse_failed']}  |  API errors: {om['api_errors']}")

    # ── SECTION 2: REGRESSION COEFFICIENT TABLE ───────────────────────────
    if reg and "error" not in reg:
        lines.append("\n── SECTION 2: REGRESSION COEFFICIENT COMPARISON ────────────────────\n")
        lines.append(
            f"  {'Attribute':<33} {'Norm tier':<10} {'Norm rank':<11} "
            f"{'LLM rank':<10} {'Norm coef':>10} {'LLM coef':>10}"
        )
        lines.append("  " + "-" * 84)
        for _, row in reg["df"].iterrows():
            lines.append(
                f"  {row['attribute']:<33} "
                f"{str(row.get('norm_tier','?')):<10} "
                f"#{int(row['norm_rank']):<10} "
                f"#{int(row['llm_rank']):<9} "
                f"{row['norm_coef']:>10.4f} "
                f"{row['llm_coef']:>10.4f}"
            )
        lines.append(f"\n  Cases used: {reg['n_cases']}")

    # ── SECTION 3: DESCRIPTIVE TIER COMPARISON ────────────────────────────
    desc = out.get("desc")
    if desc:
        lines.append("\n── SECTION 3: DESCRIPTIVE TIER COMPARISON ──────────────────────────\n")
        lines.append(
            f"  {'Attribute':<33} {'Norm':<8} {'LLM modal':<18} "
            f"{'Match':<7} {'%HIGH':<7} {'%NONE'}"
        )
        lines.append("  " + "-" * 82)
        for _, row in desc["df"].iterrows():
            match_str = "✓" if row["tier_match"] else "✗"
            lines.append(
                f"  {row['attribute']:<33} "
                f"{row['norm_tier']:<8} "
                f"{row['llm_most_common']:<18} "
                f"{match_str:<7} "
                f"{row['pct_HIGH']:<7.1f} "
                f"{row['pct_NOT_MENTIONED']:.1f}"
            )
        match_pct = desc["match_count"] / desc["total"] * 100
        lines.append(f"\n  Tier match rate : {desc['match_count']}/{desc['total']} ({match_pct:.1f}%)")
        lines.append(f"  Spearman ρ      : {desc['rho']:.3f}  p={desc['rho_pval']:.3f}  (supplementary)")

        # Over / under weighting
        over  = [(r["attribute"], r["norm_tier"], r["llm_most_common"], r["norm_rank"])
                 for _, r in desc["df"].iterrows()
                 if TIER_SCORES.get(r["llm_most_common"],0) > TIER_SCORES.get(r["norm_tier"],0)]
        under = [(r["attribute"], r["norm_tier"], r["llm_most_common"], r["norm_rank"])
                 for _, r in desc["df"].iterrows()
                 if TIER_SCORES.get(r["llm_most_common"],0) < TIER_SCORES.get(r["norm_tier"],0)]

        lines.append("\n  Over-weighted cues (LLM > normative):")
        for attr, nt, lt, nr in over:
            lines.append(f"    {attr:<33} norm={nt:<8} llm={lt:<18} (norm rank #{nr})")
        if not over:
            lines.append("    None")

        lines.append("\n  Under-weighted cues (LLM < normative):")
        for attr, nt, lt, nr in under:
            lines.append(f"    {attr:<33} norm={nt:<8} llm={lt:<18} (norm rank #{nr})")
        if not under:
            lines.append("    None")

    lines.append("\n" + "=" * W)
    text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze LLM vs normative cue weights.")
    parser.add_argument("--results",     nargs="+", required=True)
    parser.add_argument("--decoded",     default=None)
    parser.add_argument("--normative",   default="data/normative_cue_weights.csv")
    parser.add_argument("--propensities",default="data/normative_propensities.csv")
    parser.add_argument("--output",      default="analysis")
    parser.add_argument("--bootstrap-n", type=int, default=200,
                        dest="bootstrap_n")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── LOAD SHARED RESOURCES ─────────────────────────────────────────────
    if not os.path.exists(args.normative):
        raise FileNotFoundError(f"Normative weights not found: {args.normative}")
    normative = load_normative(args.normative)

    decoded_df = None
    if args.decoded:
        if not os.path.exists(args.decoded):
            print(f"Warning: decoded CSV not found — skipping regression.")
        else:
            decoded_df = load_decoded(args.decoded)

    prop_df = None
    if os.path.exists(args.propensities):
        prop_df = load_propensities(args.propensities)
    else:
        print(f"Note: propensities file not found at {args.propensities} — "
              f"propensity correlation will be skipped.\n"
              f"Run: python normative_weights.py --input data/german_credit_decoded.csv")

    # ── RUN ANALYSIS PER FILE ─────────────────────────────────────────────
    all_results   = {}   # path → raw results_df
    all_outputs   = []   # list of output dicts for comparison table

    for results_path in args.results:
        if not os.path.exists(results_path):
            print(f"Warning: {results_path} not found — skipping.")
            continue

        print(f"\nAnalysing: {results_path}")
        results_df  = load_results(results_path)
        model_label = (results_df["model"].iloc[0]
                       if "model" in results_df.columns else results_path)
        condition   = (results_df["condition"].iloc[0]
                       if "condition" in results_df.columns else "unknown")

        om   = output_metrics(results_df, model_label)
        desc = descriptive_analysis(results_df, normative, model_label)
        reg  = None
        if decoded_df is not None:
            reg = regression_analysis(results_df, decoded_df, normative,
                                      prop_df, model_label)

        out = {
            "model":     model_label,
            "condition": condition,
            "om":        om,
            "desc":      desc,
            "reg":       reg,
            "path":      results_path,
        }
        all_outputs.append(out)
        all_results[results_path] = results_df

        # Write individual report
        stem        = os.path.splitext(os.path.basename(results_path))[0]
        report_path = os.path.join(args.output, f"report_{stem}.txt")
        write_report(out, report_path)

        # Save descriptive CSV
        csv_path = os.path.join(args.output, f"descriptive_{stem}.csv")
        desc["df"].to_csv(csv_path, index=False)
        print(f"\nDescriptive CSV : {csv_path}")
        print(f"Report          : {report_path}")

    # ── MULTI-MODEL / CONDITION COMPARISON ───────────────────────────────
    if len(all_outputs) > 1:
        print("\n" + "=" * 72)
        print("CROSS-CONDITION COMPARISON")
        print("=" * 72)

        # Summary table
        header = (f"\n  {'Model/Condition':<30} {'AUC':>6} {'Acc%':>6} "
                  f"{'Kappa':>7} {'Cosine':>8} {'Pearson':>8} "
                  f"{'PropCorr':>9} {'Good%':>6}")
        print(header)
        print("  " + "-" * 90)

        for o in all_outputs:
            om  = o["om"]
            reg = o.get("reg") or {}
            label = f"{o['model'].split('/')[-1][:15]} [{o['condition']}]"
            cos  = f"{reg['cosine_sim']:.3f}"  if "cosine_sim"  in reg else "  N/A "
            pr   = f"{reg['pearson_r']:.3f}"   if "pearson_r"   in reg else "  N/A "
            pc   = f"{reg['prop_corr']:.3f}"   if reg.get("prop_corr") is not None else "  N/A "
            auc  = f"{reg['auc']:.3f}"         if "auc"         in reg else " N/A "
            print(
                f"  {label:<30} {auc:>6} {om['accuracy']*100:>6.1f} "
                f"{om['kappa']:>7.3f} {cos:>8} {pr:>8} {pc:>9} {om['good_rate']:>6.1f}"
            )

        # Bootstrap Δ cosine for each pair
        if decoded_df is not None and len(all_outputs) == 2:
            print("\n── BOOTSTRAP Δ COSINE (200 replicates) ──────────────────────────────\n")
            a, b = all_outputs[0], all_outputs[1]
            if a["reg"] and b["reg"]:
                boot = bootstrap_delta_cosine(
                    a["reg"], b["reg"], decoded_df,
                    all_results[a["path"]], all_results[b["path"]],
                    n_boot=args.bootstrap_n,
                )
                if "error" in boot:
                    print(f"  Bootstrap skipped: {boot['error']}")
                else:
                    sig = "Yes" if boot["significant"] else "No"
                    print(f"  Condition A : {a['condition']} ({a['model'].split('/')[-1]})")
                    print(f"  Condition B : {b['condition']} ({b['model'].split('/')[-1]})")
                    print(f"  Δ cosine    : {boot['observed_delta']:+.3f}")
                    print(f"  p-value     : {boot['p_value']:.3f}  (one-sided)")
                    print(f"  95% CI      : [{boot['ci_lower']:+.3f}, {boot['ci_upper']:+.3f}]")
                    print(f"  Significant : {sig}")

        # Save comparison table
        comp_path = os.path.join(args.output, "comparison_table.txt")
        with open(comp_path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            for o in all_outputs:
                om  = o["om"]
                reg = o.get("reg") or {}
                label = f"{o['model'].split('/')[-1][:15]} [{o['condition']}]"
                cos = f"{reg['cosine_sim']:.3f}" if "cosine_sim" in reg else "N/A"
                pr  = f"{reg['pearson_r']:.3f}"  if "pearson_r"  in reg else "N/A"
                pc  = f"{reg.get('prop_corr') or 'N/A'}"
                auc = f"{reg['auc']:.3f}"         if "auc"        in reg else "N/A"
                f.write(
                    f"  {label:<30} {auc:>6} {om['accuracy']*100:>6.1f} "
                    f"{om['kappa']:>7.3f} {cos:>8} {pr:>8} {pc:>9} {om['good_rate']:>6.1f}\n"
                )
        print(f"\nComparison table saved: {comp_path}")


if __name__ == "__main__":
    main()
