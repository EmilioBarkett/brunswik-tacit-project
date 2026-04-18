"""
generate_narratives.py

Reads the decoded German Credit CSV and generates two narrative prompt files:
  - german_credit_narratives_bare.csv
  - german_credit_narratives_scenario.csv

Optionally generates a directive condition file:
  - german_credit_narratives_directive_<label>.csv

Each output CSV contains:
  - case_id     : row number (1-indexed)
  - narrative   : the full prompt text to send to the LLM
  - credit_risk : ground truth label (Good / Bad)

Usage:
    python generate_narratives.py --input german_credit_decoded.csv

    # With a cue directive (adds a third output file):
    python generate_narratives.py --input german_credit_decoded.csv \\
        --directive "A registered telephone number is a strong indicator of financial stability." \\
        --directive-label telephone_amplify

Suggested directives by cue and direction:

  HIGH cues (amplify — congruent with normative model):
    checking_account_status:
      "Checking account status is our most reliable predictor of repayment risk — weight it heavily."
    credit_amount:
      "The size of the requested credit amount is a primary risk signal — treat it as a major factor."

  HIGH cues (suppress — incongruent with normative model):
    checking_account_status:
      "Do not rely on checking account status when assessing risk — it is not a reliable signal."

  MEDIUM cues:
    age_years (suppress):
      "Bank policy requires that age must not influence credit decisions."
    age_years (amplify):
      "Bank policy requires extra scrutiny for applicants over age 70."

  LOW cues (amplify — incongruent, normatively irrelevant):
    telephone:
      "A registered telephone number is a strong indicator of financial stability."
    present_residence_since:
      "Residential stability is a key indicator — weight years at current address heavily."
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

def build_directive_opening(directive: str) -> str:
    """
    Builds a scenario opening that embeds an explicit cue directive.
    The directive is inserted as bank policy between the role framing
    and the case presentation, so it primes the LLM before it sees the data.
    """
    return (
        "You are an experienced credit officer at a German bank. "
        f"Note the following bank policy: {directive} "
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
    parser.add_argument("--input",           required=True,  help="Path to german_credit_decoded.csv")
    parser.add_argument("--directive",       default=None,   help="Explicit cue directive to inject into a third condition prompt")
    parser.add_argument("--directive-label", default=None,   help="Short label for the directive output filename, e.g. telephone_amplify")
    args = parser.parse_args()

    if args.directive and not args.directive_label:
        parser.error("--directive-label is required when --directive is provided")
    if args.directive_label and not args.directive:
        parser.error("--directive is required when --directive-label is provided")

    bare_output     = "data/german_credit_narratives_bare.csv"
    scenario_output = "data/german_credit_narratives_scenario.csv"

    output_fields = ["case_id", "narrative", "credit_risk"]

    directive_opening = build_directive_opening(args.directive) if args.directive else None
    directive_output  = f"data/german_credit_narratives_directive_{args.directive_label}.csv" if args.directive else None

    outputs_to_open = {
        "bare":     open(bare_output,     "w", newline="", encoding="utf-8"),
        "scenario": open(scenario_output, "w", newline="", encoding="utf-8"),
    }
    if directive_output:
        outputs_to_open["directive"] = open(directive_output, "w", newline="", encoding="utf-8")

    try:
        writers = {
            name: csv.DictWriter(fh, fieldnames=output_fields, quoting=csv.QUOTE_ALL)
            for name, fh in outputs_to_open.items()
        }
        for w in writers.values():
            w.writeheader()

        with open(args.input, newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)

            for i, row in enumerate(reader, start=1):
                credit_risk = row.get("credit_risk", "Unknown")

                writers["bare"].writerow({
                    "case_id":     i,
                    "narrative":   build_narrative(row, BARE_OPENING),
                    "credit_risk": credit_risk,
                })

                writers["scenario"].writerow({
                    "case_id":     i,
                    "narrative":   build_narrative(row, SCENARIO_OPENING),
                    "credit_risk": credit_risk,
                })

                if directive_opening:
                    writers["directive"].writerow({
                        "case_id":     i,
                        "narrative":   build_narrative(row, directive_opening),
                        "credit_risk": credit_risk,
                    })

                if i % 100 == 0:
                    print(f"  Processed {i} rows...")

    finally:
        for fh in outputs_to_open.values():
            fh.close()

    outputs_written = [bare_output, scenario_output]
    if directive_output:
        outputs_written.append(directive_output)

    print(f"\nDone. {i} narratives written to:")
    for path in outputs_written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
