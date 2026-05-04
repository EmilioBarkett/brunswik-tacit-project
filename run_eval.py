"""
run_eval.py

Reads a narrative CSV (bare or scenario condition), sends each narrative
to OpenRouter, captures the full reasoning response, then makes a second
scoring call to extract structured cue weights from the reasoning.

Two API calls per case:
  1. Eval call   — chosen model reasons through the credit decision
  2. Scoring call — Haiku extracts cue weights from the reasoning as JSON

Usage:
    python run_eval.py --input german_credit_narratives_bare.csv --model anthropic/claude-haiku-4-5 --limit 10
    python run_eval.py --input german_credit_narratives_bare.csv --model openai/gpt-4o --limit 10
    python run_eval.py --input german_credit_narratives_bare.csv --model meta-llama/llama-3.3-70b-instruct --limit 10

Arguments:
    --input     Path to narrative CSV (required)
    --model     OpenRouter model string e.g. anthropic/claude-haiku-4-5 (required)
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

# ── LOAD ENV ──────────────────────────────────────────────────────────────────

load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_KEY:
    raise EnvironmentError(
        "OPENROUTER_API_KEY not found. "
        "Make sure your .env file contains OPENROUTER_API_KEY=your-key-here"
    )

# ── SETTINGS ──────────────────────────────────────────────────────────────────

EVAL_MAX_TOKENS    = 1024
SCORING_MAX_TOKENS = 512
RETRY_ATTEMPTS     = 3
RETRY_DELAY        = 5
CALL_DELAY         = 0.5

SCORING_MODEL = "anthropic/claude-haiku-4-5"

# ── ALL 20 ATTRIBUTE NAMES ────────────────────────────────────────────────────

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

# ── API CLIENT SETUP ──────────────────────────────────────────────────────────

def get_client():
    from openai import OpenAI
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY,
    )


def call_api(client, prompt: str, model: str, max_tokens: int) -> dict:
    """
    API call via OpenRouter (OpenAI-compatible).
    """
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.choices[0].message.content
            if text is None:
                raise ValueError("Model returned None content")
            return {
                "text":          text,
                "input_tokens":  response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "error":         "",
            }
        except Exception as e:
            print(f"    Attempt {attempt}/{RETRY_ATTEMPTS} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                print(f"    Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                return {
                    "text":          "",
                    "input_tokens":  0,
                    "output_tokens": 0,
                    "error":         str(e),
                }

# ── SCORING PROMPT BUILDER ────────────────────────────────────────────────────

def build_scoring_prompt(reasoning_text: str) -> str:
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
    if not response_text:
        return "PARSE_FAILED"
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

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run LLM eval on German Credit narratives.")
    parser.add_argument("--input",  required=True)
    parser.add_argument("--model",  required=True)
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        input_stem  = os.path.splitext(os.path.basename(args.input))[0]
        model_short = args.model.replace("/", "-").split("-")[1]
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir     = os.path.join("results", "main", "v2")
        os.makedirs(run_dir, exist_ok=True)
        args.output = os.path.join(run_dir, f"results_{input_stem}_{model_short}_{timestamp}.csv")

    scoring_model = SCORING_MODEL

    completed_ids = set()
    if os.path.exists(args.output):
        with open(args.output, newline="", encoding="utf-8") as existing:
            for row in csv.DictReader(existing):
                completed_ids.add(int(row["case_id"]))
        print(f"Resuming — {len(completed_ids)} cases already completed.")

    client    = get_client()
    input_lower = args.input.lower()
    if "introspective" in input_lower:
        condition = "introspective"
    elif "court_extern" in input_lower:
        condition = "court_extern"
    elif "scenario" in input_lower:
        condition = "scenario"
    elif "directive" in input_lower:
        condition = "directive"
    else:
        condition = "bare"

    print(f"Eval model:     {args.model}")
    print(f"Scoring model:  {scoring_model}")
    print(f"Condition:      {condition}")
    print(f"Output:         {args.output}")
    print()

    output_fields = [
        "case_id", "model", "condition", "credit_risk",
        "classification", "correct",
        "eval_input_tokens", "eval_output_tokens",
        "scoring_input_tokens", "scoring_output_tokens",
        "timestamp", "response_text", "cue_weights_json", "error",
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
        run_start   = time.time()

        for row in reader:
            case_id = int(row["case_id"])

            if case_id in completed_ids:
                continue
            if cases_limit is not None and cases_run >= cases_limit:
                break

            total_to_run = cases_limit if cases_limit is not None else 1000
            print(f"  Case {case_id} / {'all' if cases_limit is None else cases_limit}...", end=" ", flush=True)

            # ── CALL 1: EVAL ──────────────────────────────────────────────────
            eval_result = call_api(
                client, row["narrative"], args.model, EVAL_MAX_TOKENS
            )

            classification = (
                parse_classification(eval_result["text"])
                if not eval_result["error"] else "API_ERROR"
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
                    client, scoring_prompt, scoring_model, SCORING_MAX_TOKENS
                )
                cue_weights_json = (
                    parse_cue_weights(scoring_result["text"])
                    if not scoring_result["error"]
                    else json.dumps({"error": f"SCORING_API_ERROR: {scoring_result['error']}"})
                )

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

            cases_run += 1

            elapsed      = time.time() - run_start
            avg_per_case = elapsed / cases_run
            remaining    = total_to_run - cases_run
            eta_secs     = int(avg_per_case * remaining)
            eta_str      = f"{eta_secs // 60}m {eta_secs % 60}s"

            print(f"→ {classification} (truth: {row['credit_risk']}) "
                  f"{'✓' if correct is True else '✗' if correct is False else '?'}"
                  f"  |  {cases_run}/{total_to_run}  ETA {eta_str}")

            time.sleep(CALL_DELAY)

    print(f"\nDone. {cases_run} cases written to: {args.output}")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    correct_count = incorrect_count = parse_failed_count = api_error_count = scoring_failed = 0

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
    print(f"  Scoring model:   {scoring_model}")
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