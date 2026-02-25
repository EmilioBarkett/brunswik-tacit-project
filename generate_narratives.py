"""
generate_narratives.py

Reads the decoded German Credit CSV and generates two narrative prompt files:
  - german_credit_narratives_bare.csv
  - german_credit_narratives_scenario.csv

Each output CSV contains:
  - case_id     : row number (1-indexed)
  - narrative   : the full prompt text to send to the LLM
  - credit_risk : ground truth label (Good / Bad)

Usage:
    python generate_narratives.py --input german_credit_decoded.csv
"""

import csv
import argparse

# ── PROMPT TEMPLATES ──────────────────────────────────────────────────────────

BARE_OPENING = "An applicant has the following profile:"

SCENARIO_OPENING = (
    "You are an experienced credit officer at a German bank. "
    "A customer has submitted a loan application. "
    "You must assess whether they are a Good or Bad credit risk "
    "based on their profile and any relevant knowledge you bring to this assessment:"
)

CLOSING_PROMPT = (
    "Based on the profile above, assess whether this applicant is a Good or Bad credit risk. "
    "First state your classification clearly. "
    "Then explain your reasoning: which factors did you weight most heavily and why, "
    "which factors did you discount and why, and what knowledge or considerations "
    "beyond the information stated above — if any — did you draw upon to reach your decision."
)

# ── ATTRIBUTE DISPLAY NAMES ───────────────────────────────────────────────────
# Maps exact CSV column names to human-readable labels.
# Order here controls the order attributes appear in the narrative.
# All 20 attributes included — no cue weighting signals by design
# (Brunswik Lens Model: actor receives all cues with equal visual weight).

ATTRIBUTE_LABELS = {
    "checking_account_status":      "Checking account status",
    "duration_months":              "Loan duration (months)",
    "credit_history":               "Credit history",
    "purpose":                      "Loan purpose",
    "credit_amount":                "Credit amount (DM)",
    "savings_account_bonds":        "Savings / bonds",
    "employment_since":             "Employment stability",
    "installment_rate_pct":         "Installment rate (% of disposable income)",
    "personal_status_sex":          "Personal status and sex",
    "other_debtors_guarantors":     "Other debtors / guarantors",
    "present_residence_since":      "Present residence since (years)",
    "property":                     "Property ownership",
    "age_years":                    "Age (years)",
    "other_installment_plans":      "Other installment plans",
    "housing":                      "Housing",
    "num_existing_credits":         "Number of existing credits at this bank",
    "job":                          "Job / employment level",
    "num_dependents":               "Number of dependents",
    "telephone":                    "Telephone registered in applicant name",
    "foreign_worker":               "Foreign worker",
}

# ── NARRATIVE BUILDER ─────────────────────────────────────────────────────────

def build_narrative(row: dict, opening: str) -> str:
    """
    Constructs a full prompt string from a decoded CSV row.
    All 20 attributes are presented as a flat equally-weighted list.
    """
    lines = [opening, ""]

    for col, label in ATTRIBUTE_LABELS.items():
        value = row.get(col, "N/A")
        lines.append(f"- {label}: {value}")

    lines.append("")
    lines.append(CLOSING_PROMPT)

    return "\n".join(lines)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate LLM prompt narratives from decoded German Credit CSV.")
    parser.add_argument("--input", required=True, help="Path to german_credit_decoded.csv")
    args = parser.parse_args()

    bare_output     = "german_credit_narratives_bare.csv"
    scenario_output = "german_credit_narratives_scenario.csv"

    output_fields = ["case_id", "narrative", "credit_risk"]

    with open(args.input, newline="", encoding="utf-8") as infile, \
         open(bare_output,     "w", newline="", encoding="utf-8") as bare_file, \
         open(scenario_output, "w", newline="", encoding="utf-8") as scenario_file:

        reader          = csv.DictReader(infile)
        bare_writer     = csv.DictWriter(bare_file,     fieldnames=output_fields, quoting=csv.QUOTE_ALL)
        scenario_writer = csv.DictWriter(scenario_file, fieldnames=output_fields, quoting=csv.QUOTE_ALL)

        bare_writer.writeheader()
        scenario_writer.writeheader()

        for i, row in enumerate(reader, start=1):
            credit_risk = row.get("credit_risk", "Unknown")

            bare_writer.writerow({
                "case_id":     i,
                "narrative":   build_narrative(row, BARE_OPENING),
                "credit_risk": credit_risk,
            })

            scenario_writer.writerow({
                "case_id":     i,
                "narrative":   build_narrative(row, SCENARIO_OPENING),
                "credit_risk": credit_risk,
            })

            if i % 100 == 0:
                print(f"  Processed {i} rows...")

    print(f"\nDone. 1000 narratives written to:")
    print(f"  {bare_output}")
    print(f"  {scenario_output}")


if __name__ == "__main__":
    main()
