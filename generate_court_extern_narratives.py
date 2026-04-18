"""
generate_court_extern_narratives.py

Generates the court-externalized narrative CSV for the Brunswik Lens Model
experiment. The model is told the normative cue rankings (derived from logistic
regression on the German Credit Dataset) as bank policy before seeing each case.

Reads:
  - data/normative_cue_weights.csv   : ranked cue weights with signs
  - data/german_credit_decoded.csv   : decoded case profiles

Outputs:
  - data/german_credit_narratives_court_extern.csv

Usage:
    python generate_court_extern_narratives.py --input data/german_credit_decoded.csv
    python generate_court_extern_narratives.py --input data/german_credit_decoded.csv \\
        --weights data/normative_cue_weights.csv
"""

import csv
import argparse
import pandas as pd
from generate_narratives import build_narrative, CLOSING_PROMPT, ATTRIBUTE_LABELS

# ── DIRECTION DESCRIPTIONS ────────────────────────────────────────────────────
# Maps each attribute to a human-readable direction description based on the
# sign of its dominant logistic regression coefficient.
# Positive coefficient → toward Good risk; Negative → toward Bad risk.

DIRECTION_TEXT = {
    "checking_account_status":   "a favorable checking account status (positive balance, no overdraft) reduces risk",
    "installment_rate_pct":      "a higher installment burden relative to disposable income increases risk",
    "credit_amount":             "larger loan amounts increase risk",
    "duration_months":           "longer loan durations increase risk",
    "purpose":                   "loan purpose is relevant — certain purposes (e.g. new car, education) are associated with lower risk",
    "credit_history":            "a strong repayment history reduces risk",
    "savings_account_bonds":     "lower savings and bond holdings increase risk",
    "employment_since":          "longer employment tenure reduces risk",
    "other_debtors_guarantors":  "having a guarantor or co-applicant reduces risk",
    "property":                  "limited property ownership increases risk",
    "age_years":                 "older applicants tend toward lower risk",
    "personal_status_sex":       "personal status and sex have moderate predictive value",
    "num_existing_credits":      "a higher number of existing credits at this bank increases risk",
    "housing":                   "housing type has limited predictive value",
    "foreign_worker":            "foreign worker status has limited predictive value",
    "other_installment_plans":   "other active installment plans have limited predictive value",
    "num_dependents":            "a higher number of dependents slightly increases risk",
    "job":                       "higher employment level slightly reduces risk",
    "telephone":                 "registered telephone has minimal predictive value",
    "present_residence_since":   "length of current residence has minimal predictive value",
}

# ── DIRECTIVE BUILDER ─────────────────────────────────────────────────────────

def build_court_extern_opening(weights_path: str) -> str:
    """
    Reads normative_cue_weights.csv and constructs a directive prompt opening
    that injects the normative cue rankings as bank policy.

    Tiers: HIGH = strong indicators, MEDIUM = moderate, LOW = weak.
    Direction is included for each cue based on coefficient sign.
    """
    df = pd.read_csv(weights_path)

    tiers = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for _, row in df.iterrows():
        attr   = row["original_attribute"]
        tier   = str(row["weight_tier"])
        label  = ATTRIBUTE_LABELS.get(attr, attr)
        direction = DIRECTION_TEXT.get(attr, "")
        tiers[tier].append(f"{label} ({direction})")

    high_text   = "; ".join(tiers["HIGH"])
    medium_text = "; ".join(tiers["MEDIUM"])
    low_text    = "; ".join(tiers["LOW"])

    directive = (
        f"Strong indicators of credit risk (weight these heavily): {high_text}. "
        f"Moderate indicators (consider these as contributing factors): {medium_text}. "
        f"Weak indicators (these have limited predictive value): {low_text}."
    )

    return (
        "You are an experienced credit officer at a German bank. "
        f"Note the following bank policy based on our credit risk model: {directive} "
        "A customer has submitted a loan application. "
        "You must assess whether they are a Good or Bad credit risk "
        "based on their profile and any relevant knowledge you bring to this assessment:"
    )

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate court-externalized narratives from normative cue weights."
    )
    parser.add_argument("--input",   required=True, help="Path to german_credit_decoded.csv")
    parser.add_argument("--weights", default="data/normative_cue_weights.csv",
                        help="Path to normative_cue_weights.csv (default: data/normative_cue_weights.csv)")
    args = parser.parse_args()

    output_path   = "data/german_credit_narratives_court_extern.csv"
    output_fields = ["case_id", "narrative", "credit_risk"]

    opening = build_court_extern_opening(args.weights)

    print("── Court-Externalized Directive ──────────────────────────────────────")
    print(opening)
    print()

    with open(args.input,  newline="", encoding="utf-8") as infile, \
         open(output_path, "w",        newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=output_fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for i, row in enumerate(reader, start=1):
            writer.writerow({
                "case_id":     i,
                "narrative":   build_narrative(row, opening),
                "credit_risk": row.get("credit_risk", "Unknown"),
            })
            if i % 100 == 0:
                print(f"  Processed {i} rows...")

    print(f"\nDone. {i} narratives written to: {output_path}")


if __name__ == "__main__":
    main()
