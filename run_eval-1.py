"""
run_eval.py

Reads a narrative CSV (bare or scenario condition), sends each narrative
to the Anthropic API, captures the full reasoning response, then makes a
second scoring call to extract structured cue weights from the reasoning.
All results are saved to a CSV for later analysis.

Two API calls per case:
  1. Eval call   — eval model (e.g. Sonnet) reasons through the credit decision
  2. Scoring call — Haiku extracts cue weights from the reasoning as JSON

Usage:
    python run_eval.py --input german_credit_narratives_bare.csv --model claude-haiku-4-5-20251001 --limit 50
    python run_eval.py --input german_credit_narratives_scenario.csv --model claude-sonnet-4-6 --limit 100

Arguments:
    --input     Path to narrative CSV (required)
    --model     Anthropic model string for eval call (required)
    --limit     Number of cases to run (optional, default: all)
    --output    Path to results CSV (optional, default: auto-generated)
"""

import csv
import argparse
import os
import time
import re
import json
from datetime import datetime
from dotenv import load_dotenv
import anthropic

# ── LOAD ENV ──────────────────────────────────────────────────────────────────

load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not API_KEY:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not found. "
        "Make sure your .env file exists and contains ANTHROPIC_API_KEY=your-key-here"
    )

# ── SETTINGS ──────────────────────────────────────────────────────────────────

EVAL_MAX_TOKENS     = 1024  # enough for full reasoning response
SCORING_MAX_TOKENS  = 512   # scoring call only needs to return JSON
RETRY_ATTEMPTS      = 3
RETRY_DELAY         = 5     # seconds between retries
CALL_DELAY          = 0.5   # seconds between cases (rate limit buffer)
SCORING_MODEL       = "claude-haiku-4-5-20251001"  # always Haiku for scoring

# ── ALL 20 ATTRIBUTE NAMES ────────────────────────────────────────────────────
# Must match column names in the decoded CSV exactly.
# Used to build the scoring prompt and validate JSON output.

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

VALID_WEIGHT_TIERS = {"HIGH", "MEDIUM", "LOW", "DISCOUNTED", "NOT_MENTIONED"}

# ── SCORING PROMPT BUILDER ────────────────────────────────────────────────────

def build_scoring_prompt(reasoning_text: str) -> str:
    """
    Builds the prompt for the second (scoring) API call.
    Asks Haiku to extract how each attribute was weighted in the reasoning.
    """
    attribute_list = "\n".join(f"- {attr}" for attr in ALL_ATTRIBUTES)

    return f"""Below is a credit officer's reasoning about a loan application.

Your task is to classify how the officer weighted each of the following 20 attributes in their reasoning.

For each attribute, assign exactly one of these tiers:
- HIGH         : explicitly identified as a major factor driving the decision
- MEDIUM       : mentioned as a contributing factor but not a primary driver
- LOW          : mentioned briefly or in passing with little influence
- DISCOUNTED   : explicitly down-weighted or dismissed by the officer
- NOT_MENTIONED: not referenced at all in the reasoning

Attributes to classify:
{attribute_list}

Return your answer as a single JSON object only. No preamble, no explanation, no markdown.
The JSON must contain exactly these 20 keys with one of the five tier values each.

Reasoning to analyse:
\"\"\"
{reasoning_text}
\"\"\"
"""

# ── CLASSIFICATION PARSER ─────────────────────────────────────────────────────

def parse_classification(response_text: str) -> str:
    """
    Extracts Good or Bad from the eval response.
    Checks the first 300 chars first, falls back to full text.
    Returns PARSE_FAILED if neither found.
    """
    lead = response_text[:300]
    match = re.search(r'\b(Good|Bad)\b', lead, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    match = re.search(r'\b(Good|Bad)\b', response_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    return "PARSE_FAILED"

# ── CUE WEIGHT PARSER ────────────────────────────────────────────────────────

def parse_cue_weights(scoring_response_text: str) -> str:
    """
    Parses the JSON cue weights from the scoring response.
    Validates that all 20 attributes are present with valid tier values.
    Returns a JSON string on success, or a SCORING_FAILED JSON string on error.
    """
    try:
        clean = scoring_response_text.strip()
        clean = re.sub(r"^```json|^```|```$", "", clean, flags=re.MULTILINE).strip()

        weights = json.loads(clean)

        missing = [a for a in ALL_ATTRIBUTES if a not in weights]
        if missing:
            return json.dumps({"error": f"SCORING_FAILED — missing keys: {missing}"})

        invalid = {k: v for k, v in weights.items() if v not in VALID_WEIGHT_TIERS}
        if invalid:
            return json.dumps({"error": f"SCORING_FAILED — invalid tiers: {invalid}"})

        return json.dumps(weights)

    except json.JSONDecodeError as e:
        return json.dumps({"error": f"SCORING_FAILED — JSON parse error: {str(e)}"})

# ── API CALL WITH RETRY ───────────────────────────────────────────────────────

def call_api(client: anthropic.Anthropic, prompt: str, model: str, max_tokens: int) -> dict:
    """
    Generic API call with retry logic.
    Returns response text, token counts, and any error.
    """
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return {
                "text":           response.content[0].text,
                "input_tokens":   response.usage.input_tokens,
                "output_tokens":  response.usage.output_tokens,
                "error":          "",
            }
        except Exception as e:
            print(f"    Attempt {attempt}/{RETRY_ATTEMPTS} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                print(f"    Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                return {
                    "text":           "",
                    "input_tokens":   0,
                    "output_tokens":  0,
                    "error":          str(e),
                }

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run LLM eval on German Credit narratives.")
    parser.add_argument("--input",  required=True,  help="Path to narrative CSV")
    parser.add_argument("--model",  required=True,  help="Anthropic model for eval call")
    parser.add_argument("--limit",  type=int, default=None, help="Max cases to run (default: all)")
    parser.add_argument("--output", default=None,   help="Path to results CSV (default: auto-generated)")
    args = parser.parse_args()

    # Auto-generate output filename
    if args.output is None:
        input_stem  = os.path.splitext(os.path.basename(args.input))[0]
        model_short = args.model.split("-")[1]
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results_{input_stem}_{model_short}_{timestamp}.csv"

    # Resume support
    completed_ids = set()
    if os.path.exists(args.output):
        with open(args.output, newline="", encoding="utf-8") as existing:
            for row in csv.DictReader(existing):
                completed_ids.add(int(row["case_id"]))
        print(f"Resuming — {len(completed_ids)} cases already completed.")

    client    = anthropic.Anthropic(api_key=API_KEY)
    condition = "scenario" if "scenario" in args.input.lower() else "bare"

    output_fields = [
        "case_id",
        "model",
        "condition",
        "credit_risk",
        "classification",
        "correct",
        "eval_input_tokens",
        "eval_output_tokens",
        "scoring_input_tokens",
        "scoring_output_tokens",
        "timestamp",
        "response_text",
        "cue_weights_json",
        "error",
    ]

    write_mode = "a" if completed_ids else "w"

    with open(args.input, newline="", encoding="utf-8") as infile, \
         open(args.output, write_mode, newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=output_fields, quoting=csv.QUOTE_ALL)

        if not completed_ids:
            writer.writeheader()

        cases_run   = 0
        cases_limit = args.limit

        for row in reader:
            case_id = int(row["case_id"])

            if case_id in completed_ids:
                continue
            if cases_limit is not None and cases_run >= cases_limit:
                break

            print(f"  Case {case_id} / {'all' if cases_limit is None else cases_limit}...", end=" ", flush=True)

            # ── CALL 1: EVAL ──────────────────────────────────────────────────
            eval_result = call_api(
                client, row["narrative"], args.model, EVAL_MAX_TOKENS
            )

            classification = (
                parse_classification(eval_result["text"])
                if not eval_result["error"]
                else "API_ERROR"
            )

            correct = (
                classification.lower() == row["credit_risk"].lower()
                if classification not in ("PARSE_FAILED", "API_ERROR")
                else "UNKNOWN"
            )

            # ── CALL 2: SCORING ───────────────────────────────────────────────
            cue_weights_json = json.dumps({"error": "SKIPPED — eval failed"})
            scoring_result   = {"input_tokens": 0, "output_tokens": 0}

            if eval_result["text"] and not eval_result["error"]:
                scoring_prompt = build_scoring_prompt(eval_result["text"])
                scoring_result = call_api(
                    client, scoring_prompt, SCORING_MODEL, SCORING_MAX_TOKENS
                )
                cue_weights_json = (
                    parse_cue_weights(scoring_result["text"])
                    if not scoring_result["error"]
                    else json.dumps({"error": f"SCORING_API_ERROR: {scoring_result['error']}"})
                )

            # ── WRITE ROW ─────────────────────────────────────────────────────
            writer.writerow({
                "case_id":               case_id,
                "model":                 args.model,
                "condition":             condition,
                "credit_risk":           row["credit_risk"],
                "classification":        classification,
                "correct":               correct,
                "eval_input_tokens":     eval_result["input_tokens"],
                "eval_output_tokens":    eval_result["output_tokens"],
                "scoring_input_tokens":  scoring_result["input_tokens"],
                "scoring_output_tokens": scoring_result["output_tokens"],
                "timestamp":             datetime.now().isoformat(),
                "response_text":         eval_result["text"],
                "cue_weights_json":      cue_weights_json,
                "error":                 eval_result["error"],
            })
            outfile.flush()

            print(f"→ {classification} (truth: {row['credit_risk']}) {'✓' if correct is True else '✗' if correct is False else '?'}")

            cases_run += 1
            time.sleep(CALL_DELAY)

    print(f"\nDone. {cases_run} cases written to: {args.output}")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    correct_count      = 0
    incorrect_count    = 0
    parse_failed_count = 0
    api_error_count    = 0
    scoring_failed     = 0

    with open(args.output, newline="", encoding="utf-8") as results_file:
        for result_row in csv.DictReader(results_file):
            if result_row["classification"] == "API_ERROR":
                api_error_count += 1
            elif result_row["classification"] == "PARSE_FAILED":
                parse_failed_count += 1
            elif result_row["correct"] == "True":
                correct_count += 1
            else:
                incorrect_count += 1

            try:
                w = json.loads(result_row["cue_weights_json"])
                if "error" in w:
                    scoring_failed += 1
            except Exception:
                scoring_failed += 1

    scoreable = correct_count + incorrect_count
    accuracy  = (correct_count / scoreable * 100) if scoreable > 0 else 0

    print(f"\n── RESULTS SUMMARY ──────────────────────────────")
    print(f"  Eval model:      {args.model}")
    print(f"  Scoring model:   {SCORING_MODEL}")
    print(f"  Condition:       {condition}")
    print(f"  Cases run:       {cases_run}")
    print(f"  Correct:         {correct_count}")
    print(f"  Incorrect:       {incorrect_count}")
    print(f"  Accuracy:        {accuracy:.1f}%")
    if parse_failed_count:
        print(f"  Parse failed:    {parse_failed_count} (manual review needed)")
    if api_error_count:
        print(f"  API errors:      {api_error_count} (rerun to retry)")
    if scoring_failed:
        print(f"  Scoring failed:  {scoring_failed} (cue weight extraction failed)")
    print(f"─────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
