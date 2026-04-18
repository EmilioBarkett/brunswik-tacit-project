"""
normative_weights.py

Runs a logistic regression on the decoded German Credit Dataset to derive
normative cue weights. These weights form the left side of the Brunswik Lens
Model — the empirically correct way to weight each cue to predict credit risk.

Outputs:
  - normative_cue_weights.csv      : cue weights ranked by importance
  - normative_weights_report.txt   : human-readable summary for the paper
  - normative_propensities.csv     : per-case predicted probabilities (Good) from the
                                     fitted model, used for propensity correlation in
                                     analyze_weights.py

Usage:
    python normative_weights.py --input german_credit_decoded.csv
"""

import csv
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Derive normative cue weights via logistic regression.")
    parser.add_argument("--input", required=True, help="Path to german_credit_decoded.csv")
    args = parser.parse_args()

    # ── LOAD DATA ─────────────────────────────────────────────────────────────
    df = pd.read_csv(args.input)

    print(f"Loaded {len(df)} cases.")
    print(f"Class distribution:\n{df['credit_risk'].value_counts().to_string()}\n")

    # ── ENCODE TARGET ─────────────────────────────────────────────────────────
    df["target"] = (df["credit_risk"] == "Good").astype(int)  # Good=1, Bad=0

    # ── ENCODE FEATURES ───────────────────────────────────────────────────────
    # One-hot encode all categorical columns, leave numerical as-is
    feature_cols = [c for c in df.columns if c not in ("credit_risk", "target")]

    X_raw = df[feature_cols].copy()

    # Identify categorical vs numerical columns
    categorical_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols   = X_raw.select_dtypes(include=["number"]).columns.tolist()

    print(f"Numerical attributes  ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical attributes({len(categorical_cols)}): {categorical_cols}\n")

    # One-hot encode categoricals
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False)
    y = df["target"].values

    feature_names = X_encoded.columns.tolist()

    # ── FIT LOGISTIC REGRESSION ───────────────────────────────────────────────
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=1000, random_state=42, C=1.0))
    ])

    pipeline.fit(X_encoded, y)

    # Cross-validated accuracy (5-fold)
    cv_scores = cross_val_score(pipeline, X_encoded, y, cv=5, scoring="accuracy")
    print(f"Logistic Regression 5-fold CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})\n")

    # ── EXTRACT COEFFICIENTS ──────────────────────────────────────────────────
    # Coefficients after scaling represent relative importance of each feature
    lr_model = pipeline.named_steps["lr"]
    coefficients = lr_model.coef_[0]

    coef_df = pd.DataFrame({
        "feature":          feature_names,
        "coefficient":      coefficients,
        "abs_coefficient":  np.abs(coefficients),
    }).sort_values("abs_coefficient", ascending=False)

    # ── MAP BACK TO ORIGINAL 20 ATTRIBUTES ───────────────────────────────────
    # Each original attribute may have multiple one-hot columns.
    # We aggregate by taking the max absolute coefficient per original attribute.
    def get_original_attr(feature_name):
        for col in feature_cols:
            if feature_name == col or feature_name.startswith(col + "_"):
                return col
        return feature_name

    coef_df["original_attribute"] = coef_df["feature"].apply(get_original_attr)

    # For each original attribute, find the one-hot column with the largest
    # absolute coefficient and preserve its signed coefficient.
    idx_max = coef_df.groupby("original_attribute")["abs_coefficient"].idxmax()
    attr_weights = coef_df.loc[idx_max, ["original_attribute", "coefficient", "abs_coefficient"]].copy()
    attr_weights = attr_weights.rename(columns={"abs_coefficient": "normative_weight"})
    attr_weights = attr_weights.sort_values("normative_weight", ascending=False).reset_index(drop=True)

    attr_weights["rank"]           = attr_weights.index + 1
    attr_weights["weight_tier"]    = pd.cut(
        attr_weights["normative_weight"],
        bins=3,
        labels=["LOW", "MEDIUM", "HIGH"]
    )

    # ── PROPENSITY SCORES ────────────────────────────────────────────────────
    # Cross-validated predicted probabilities of Good (class=1) per case.
    # Using cross_val_predict so propensities are out-of-fold (not in-sample),
    # matching the approach used in analyze_weights.py for LLM propensities.
    norm_proba = cross_val_predict(pipeline, X_encoded, y, cv=5, method="predict_proba")
    propensity_df = pd.DataFrame({
        "case_id":            range(1, len(df) + 1),
        "normative_prob_good": norm_proba[:, 1],   # P(Good) per case
        "credit_risk":        df["credit_risk"].values,
    })

    # ── SAVE OUTPUTS ─────────────────────────────────────────────────────────
    attr_weights[["original_attribute", "coefficient", "normative_weight", "rank", "weight_tier"]].to_csv("data/normative_cue_weights.csv", index=False)
    print("Saved: data/normative_cue_weights.csv")

    propensity_df.to_csv("data/normative_propensities.csv", index=False)
    print("Saved: data/normative_propensities.csv\n")

    # ── PRINT REPORT ─────────────────────────────────────────────────────────
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("NORMATIVE CUE WEIGHTS — GERMAN CREDIT DATASET")
    report_lines.append("Logistic Regression (standardized coefficients)")
    report_lines.append("=" * 60)
    report_lines.append(f"Model accuracy (5-fold CV): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    report_lines.append(f"N cases: {len(df)}")
    report_lines.append("")
    report_lines.append(f"{'Rank':<6} {'Attribute':<35} {'Weight':<10} {'Tier'}")
    report_lines.append("-" * 60)

    for _, row in attr_weights.iterrows():
        report_lines.append(
            f"{int(row['rank']):<6} {row['original_attribute']:<35} "
            f"{row['normative_weight']:.4f}     {row['weight_tier']}"
        )

    report_lines.append("")
    report_lines.append("INTERPRETATION")
    report_lines.append("-" * 60)
    report_lines.append("HIGH tier   : Cues the normative model relies on most.")
    report_lines.append("              LLM should weight these heavily.")
    report_lines.append("MEDIUM tier : Moderately predictive cues.")
    report_lines.append("LOW tier    : Weakly predictive cues.")
    report_lines.append("              LLM over-weighting these = tacit knowledge failure.")
    report_lines.append("")
    report_lines.append("Use these weights to score LLM reasoning in run_eval.py.")
    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)
    print(report_text)

    with open("data/normative_weights_report.txt", "w") as f:
        f.write(report_text)
    print("\nSaved: data/normative_weights_report.txt")


if __name__ == "__main__":
    main()
