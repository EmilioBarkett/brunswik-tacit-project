"""
generate_introspective_narratives.py

Generates the introspective-externalized narrative CSV for a specific model.
Each model receives a personalised directive that:
  (1) Feeds back which cues it over- and under-weighted in the baseline (scenario)
      condition relative to the normative logistic regression model.
  (2) Corrects for the bad-bias problem by stating the 70% Good base rate.

Deviation profiles are derived from Section 3 of each model's
report_results_german_credit_narratives_scenario_*.txt analysis report.

Reads:
  - data/german_credit_decoded.csv   : decoded case profiles

Outputs (one file per model):
  - data/german_credit_narratives_introspective_{model_key}.csv

Usage:
    python generate_introspective_narratives.py --model-key haiku \\
        --input data/german_credit_decoded.csv

    python generate_introspective_narratives.py --model-key gpt_mini \\
        --input data/german_credit_decoded.csv

Available model keys:
    haiku       → anthropic/claude-haiku-4-5       (baseline cosine 0.452)
    gpt_mini    → openai/gpt-5.4-mini              (baseline cosine 0.118)
    gpt_nano    → openai/gpt-5.4-nano              (baseline cosine 0.609)
    grok        → x-ai/grok-4.1-fast               (baseline cosine -0.283)
    deepseek    → deepseek/deepseek-v3.2            (baseline cosine 0.460)
"""

import csv
import argparse
from generate_narratives import build_narrative, CLOSING_PROMPT, ATTRIBUTE_LABELS

# ── MODEL DEVIATION PROFILES ──────────────────────────────────────────────────
# Derived from Section 3 (descriptive tier comparison) of each model's
# scenario-condition analysis report. Entries list (attribute, norm_tier, llm_tier)
# tuples for each deviation direction.
#
# ATTRIBUTE_LABELS maps internal names to display labels for the prompt text.

MODEL_PROFILES = {

    "haiku": {
        "model_id":     "anthropic/claude-haiku-4-5",
        "baseline_cosine": 0.452,
        "over_weighted": [
            # (attribute, normative_tier, llm_tier)
            ("credit_history",          "MEDIUM", "HIGH"),
            ("savings_account_bonds",   "MEDIUM", "HIGH"),
            ("foreign_worker",          "LOW",    "HIGH"),
            ("num_existing_credits",    "LOW",    "MEDIUM"),
            ("housing",                 "LOW",    "MEDIUM"),
            ("num_dependents",          "LOW",    "MEDIUM"),
            ("job",                     "LOW",    "MEDIUM"),
            ("present_residence_since", "LOW",    "MEDIUM"),
        ],
        "under_weighted": [
            ("installment_rate_pct",       "HIGH",   "MEDIUM"),
            ("credit_amount",              "HIGH",   "MEDIUM"),
            ("duration_months",            "HIGH",   "MEDIUM"),
            ("other_debtors_guarantors",   "MEDIUM", "NOT_MENTIONED"),
            ("age_years",                  "MEDIUM", "NOT_MENTIONED"),
            ("personal_status_sex",        "MEDIUM", "NOT_MENTIONED"),
            ("other_installment_plans",    "LOW",    "NOT_MENTIONED"),
        ],
    },

    "gpt_mini": {
        "model_id":     "openai/gpt-5.4-mini",
        "baseline_cosine": 0.118,
        "over_weighted": [
            ("credit_history",          "MEDIUM", "HIGH"),
            ("employment_since",        "MEDIUM", "HIGH"),
            ("job",                     "LOW",    "HIGH"),
            ("num_existing_credits",    "LOW",    "MEDIUM"),
            ("housing",                 "LOW",    "MEDIUM"),
            ("foreign_worker",          "LOW",    "MEDIUM"),
            ("other_installment_plans", "LOW",    "MEDIUM"),
            ("present_residence_since", "LOW",    "MEDIUM"),
        ],
        "under_weighted": [
            ("checking_account_status",  "HIGH",   "MEDIUM"),
            ("credit_amount",            "HIGH",   "MEDIUM"),
            ("duration_months",          "HIGH",   "MEDIUM"),
            ("other_debtors_guarantors", "MEDIUM", "NOT_MENTIONED"),
            ("personal_status_sex",      "MEDIUM", "DISCOUNTED"),
        ],
    },

    "gpt_nano": {
        "model_id":     "openai/gpt-5.4-nano",
        "baseline_cosine": 0.609,
        "over_weighted": [
            ("credit_history",          "MEDIUM", "HIGH"),
            ("savings_account_bonds",   "MEDIUM", "HIGH"),
            ("num_existing_credits",    "LOW",    "MEDIUM"),
            ("housing",                 "LOW",    "MEDIUM"),
            ("job",                     "LOW",    "MEDIUM"),
            ("present_residence_since", "LOW",    "MEDIUM"),
        ],
        "under_weighted": [
            ("credit_amount",           "HIGH",  "NOT_MENTIONED"),
            ("duration_months",         "HIGH",  "NOT_MENTIONED"),
            ("personal_status_sex",     "MEDIUM","NOT_MENTIONED"),
            ("other_installment_plans", "LOW",   "NOT_MENTIONED"),
            ("num_dependents",          "LOW",   "NOT_MENTIONED"),
            ("telephone",               "LOW",   "NOT_MENTIONED"),
        ],
    },

    "grok": {
        "model_id":     "x-ai/grok-4.1-fast",
        "baseline_cosine": -0.283,
        "over_weighted": [
            ("credit_history",          "MEDIUM", "HIGH"),
            ("savings_account_bonds",   "MEDIUM", "HIGH"),
            ("employment_since",        "MEDIUM", "HIGH"),
            ("num_existing_credits",    "LOW",    "HIGH"),
            ("foreign_worker",          "LOW",    "HIGH"),
            ("job",                     "LOW",    "HIGH"),
            ("present_residence_since", "LOW",    "MEDIUM"),
        ],
        "under_weighted": [
            ("installment_rate_pct",     "HIGH",   "DISCOUNTED"),
            ("credit_amount",            "HIGH",   "LOW"),
            ("duration_months",          "HIGH",   "LOW"),
            ("purpose",                  "MEDIUM", "LOW"),
            ("other_debtors_guarantors", "MEDIUM", "LOW"),
            ("property",                 "MEDIUM", "DISCOUNTED"),
            ("age_years",                "MEDIUM", "LOW"),
            ("personal_status_sex",      "MEDIUM", "LOW"),
            ("housing",                  "LOW",    "DISCOUNTED"),
        ],
    },

    "deepseek": {
        "model_id":     "deepseek/deepseek-v3.2",
        "baseline_cosine": 0.460,
        "over_weighted": [
            ("savings_account_bonds",   "MEDIUM", "HIGH"),
            ("foreign_worker",          "LOW",    "HIGH"),
            ("num_existing_credits",    "LOW",    "MEDIUM"),
            ("housing",                 "LOW",    "MEDIUM"),
            ("num_dependents",          "LOW",    "MEDIUM"),
            ("job",                     "LOW",    "MEDIUM"),
        ],
        "under_weighted": [
            ("installment_rate_pct",       "HIGH",   "MEDIUM"),
            ("credit_amount",              "HIGH",   "MEDIUM"),
            ("duration_months",            "HIGH",   "MEDIUM"),
            ("other_debtors_guarantors",   "MEDIUM", "NOT_MENTIONED"),
            ("age_years",                  "MEDIUM", "NOT_MENTIONED"),
            ("personal_status_sex",        "MEDIUM", "NOT_MENTIONED"),
            ("other_installment_plans",    "LOW",    "NOT_MENTIONED"),
            ("present_residence_since",    "LOW",    "NOT_MENTIONED"),
        ],
    },
}

# ── TIER LABELS ───────────────────────────────────────────────────────────────

TIER_LABELS = {
    "HIGH":          "a primary factor",
    "MEDIUM":        "a contributing factor",
    "LOW":           "a minor factor",
    "DISCOUNTED":    "actively discounted",
    "NOT_MENTIONED": "not considered",
}

# ── DIRECTIVE BUILDER ─────────────────────────────────────────────────────────

def build_introspective_opening(model_key: str) -> str:
    """
    Constructs a model-specific introspective-externalized directive.

    The directive contains three elements:
      1. Role framing (experienced credit officer).
      2. Calibration feedback: which cues the model over- vs under-weights
         relative to the normative logistic regression benchmark.
      3. Base rate correction: historically ~70% of applicants are Good risks.
    """
    profile    = MODEL_PROFILES[model_key]
    over_list  = profile["over_weighted"]
    under_list = profile["under_weighted"]

    def fmt_over(attr, norm_tier, llm_tier):
        label    = ATTRIBUTE_LABELS.get(attr, attr)
        norm_lbl = TIER_LABELS.get(norm_tier, norm_tier.lower())
        llm_lbl  = TIER_LABELS.get(llm_tier,  llm_tier.lower())
        return (
            f"{label} (normatively: {norm_lbl}; "
            f"you have been treating it as {llm_lbl} — reduce its weight)"
        )

    def fmt_under(attr, norm_tier, llm_tier):
        label    = ATTRIBUTE_LABELS.get(attr, attr)
        norm_lbl = TIER_LABELS.get(norm_tier, norm_tier.lower())
        llm_lbl  = TIER_LABELS.get(llm_tier,  llm_tier.lower())
        return (
            f"{label} (normatively: {norm_lbl}; "
            f"you have been treating it as {llm_lbl} — increase its weight)"
        )

    over_text  = "; ".join(fmt_over(*t) for t in over_list)
    under_text = "; ".join(fmt_under(*t) for t in under_list)

    calibration = (
        f"Calibration feedback for your decision policy: "
        f"Based on prior analysis, you tend to over-weight the following factors "
        f"relative to the normative benchmark — please reduce your reliance on them: "
        f"{over_text}. "
        f"Conversely, you tend to under-weight these factors — "
        f"please give them more consideration: {under_text}. "
    )

    base_rate = (
        "Base rate correction: "
        "Historically, approximately 70% of loan applicants at this bank have been "
        "classified as Good credit risks. Your approval rate should reflect this baseline — "
        "if you are classifying substantially fewer than 70% of applicants as Good, "
        "you are likely being over-cautious. "
    )

    return (
        "You are an experienced credit officer at a German bank. "
        f"{calibration}"
        f"{base_rate}"
        "A customer has submitted a loan application. "
        "You must assess whether they are a Good or Bad credit risk "
        "based on their profile and the calibration guidance above:"
    )

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate introspective-externalized narratives for a specific model."
    )
    parser.add_argument(
        "--model-key", required=True,
        choices=list(MODEL_PROFILES.keys()),
        help="Model identifier: haiku | gpt_mini | gpt_nano | grok | deepseek"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to german_credit_decoded.csv"
    )
    args = parser.parse_args()

    model_key   = args.model_key
    profile     = MODEL_PROFILES[model_key]
    output_path = f"data/german_credit_narratives_introspective_{model_key}.csv"

    opening = build_introspective_opening(model_key)

    print(f"── Introspective Directive for [{model_key}] ({profile['model_id']}) ───")
    print(f"  Baseline cosine: {profile['baseline_cosine']}")
    print(f"  Over-weighted cues:  {len(profile['over_weighted'])}")
    print(f"  Under-weighted cues: {len(profile['under_weighted'])}")
    print()
    print("Opening directive (truncated to 500 chars):")
    print(opening[:500] + ("..." if len(opening) > 500 else ""))
    print()

    output_fields = ["case_id", "narrative", "credit_risk"]

    with open(args.input,  newline="", encoding="utf-8") as infile, \
         open(output_path, "w",        newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=output_fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        count = 0
        for i, row in enumerate(reader, start=1):
            writer.writerow({
                "case_id":     i,
                "narrative":   build_narrative(row, opening),
                "credit_risk": row.get("credit_risk", "Unknown"),
            })
            count = i
            if i % 100 == 0:
                print(f"  Processed {i} rows...")

    print(f"\nDone. {count} narratives written to: {output_path}")
    print(f"Run eval with:")
    print(f"  python run_eval.py \\")
    print(f"    --input {output_path} \\")
    print(f"    --model {profile['model_id']}")


if __name__ == "__main__":
    main()
