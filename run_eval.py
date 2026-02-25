"""
run_eval.py

Reads a narrative CSV (bare or scenario condition), sends each narrative
to the Anthropic API, and saves the full response plus parsed classification
to a results CSV for later analysis.

Usage:
    python run_eval.py --input german_credit_narratives_bare.csv --model claude-haiku-4-5-20251001 --limit 50
    python run_eval.py --input german_credit_narratives_scenario.csv --model claude-sonnet-4-6 --limit 100

Arguments:
    --input     Path to narrative CSV (required)
    --model     Anthropic model string (required)
    --limit     Number of cases to run (optional, default: all)
    --output    Path to results CSV (optional, default: auto-generated from input name)
"""

import csv
import argparse
import os
import time
import re
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

MAX_TOKENS      = 1024   # enough for a full reasoning response
RETRY_ATTEMPTS  = 3      # number of retries on API failure
RETRY_DELAY     = 5      # seconds to wait between retries
CALL_DELAY      = 0.5    # seconds to wait between successful calls (rate limit buffer)

# ── CLASSIFICATION PARSER ─────────────────────────────────────────────────────

def parse_classification(response_text: str) -> str:
    """
    Attempts to extract Good or Bad from the model's response.
    Looks in the first 3 sentences first, then falls back to full text.
    Returns "PARSE_FAILED" if neither is found — these cases get manual review.
    """
    # Check first 300 characters first (classification should come early)
    lead = response_text[:300]

    # Look for "Good" or "Bad" as standalone words (case-insensitive)
    match = re.search(r'\b(Good|Bad)\b', lead, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    # Fall back to full response
    match = re.search(r'\b(Good|Bad)\b', response_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    return "PARSE_FAILED"

# ── API CALL WITH RETRY ───────────────────────────────────────────────────────

def call_api(client: anthropic.Anthropic, narrative: str, model: str) -> dict:
    """
    Sends a single narrative to the API and returns a result dict.
    Retries up to RETRY_ATTEMPTS times on failure.
    """
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "user", "content": narrative}
                ]
            )
            response_text = response.content[0].text
            return {
                "response_text":    response_text,
                "classification":   parse_classification(response_text),
                "input_tokens":     response.usage.input_tokens,
                "output_tokens":    response.usage.output_tokens,
                "error":            "",
            }

        except Exception as e:
            print(f"    Attempt {attempt}/{RETRY_ATTEMPTS} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                print(f"    Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                return {
                    "response_text":    "",
                    "classification":   "API_ERROR",
                    "input_tokens":     0,
                    "output_tokens":    0,
                    "error":            str(e),
                }

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run LLM eval on German Credit narratives.")
    parser.add_argument("--input",  required=True,  help="Path to narrative CSV")
    parser.add_argument("--model",  required=True,  help="Anthropic model string e.g. claude-haiku-4-5-20251001")
    parser.add_argument("--limit",  type=int, default=None, help="Max number of cases to run (default: all)")
    parser.add_argument("--output", default=None,   help="Path to results CSV (default: auto-generated)")
    args = parser.parse_args()

    # Auto-generate output filename from input name and model
    if args.output is None:
        input_stem  = os.path.splitext(os.path.basename(args.input))[0]
        model_short = args.model.split("-")[1]  # e.g. "haiku" from "claude-haiku-..."
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results_{input_stem}_{model_short}_{timestamp}.csv"

    # Check for existing results to enable resuming
    completed_ids = set()
    if os.path.exists(args.output):
        with open(args.output, newline="", encoding="utf-8") as existing:
            reader = csv.DictReader(existing)
            for row in reader:
                completed_ids.add(int(row["case_id"]))
        print(f"Resuming — {len(completed_ids)} cases already completed.")

    # Init Anthropic client
    client = anthropic.Anthropic(api_key=API_KEY)

    output_fields = [
        "case_id",
        "model",
        "condition",
        "credit_risk",
        "classification",
        "correct",
        "input_tokens",
        "output_tokens",
        "timestamp",
        "response_text",
        "error",
    ]

    # Derive condition label from input filename
    condition = "scenario" if "scenario" in args.input.lower() else "bare"

    # Open input and output files
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

            # Skip already completed cases (resume support)
            if case_id in completed_ids:
                continue

            # Stop if limit reached
            if cases_limit is not None and cases_run >= cases_limit:
                break

            print(f"  Case {case_id} / {'all' if cases_limit is None else cases_limit}...", end=" ")

            result = call_api(client, row["narrative"], args.model)

            correct = (
                result["classification"].lower() == row["credit_risk"].lower()
                if result["classification"] not in ("PARSE_FAILED", "API_ERROR")
                else "UNKNOWN"
            )

            writer.writerow({
                "case_id":        case_id,
                "model":          args.model,
                "condition":      condition,
                "credit_risk":    row["credit_risk"],
                "classification": result["classification"],
                "correct":        correct,
                "input_tokens":   result["input_tokens"],
                "output_tokens":  result["output_tokens"],
                "timestamp":      datetime.now().isoformat(),
                "response_text":  result["response_text"],
                "error":          result["error"],
            })
            outfile.flush()  # write each row immediately so progress is saved

            print(f"→ {result['classification']} (truth: {row['credit_risk']}) {'✓' if correct is True else '✗' if correct is False else '?'}")

            cases_run += 1
            time.sleep(CALL_DELAY)

    print(f"\nDone. {cases_run} cases written to: {args.output}")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    correct_count       = 0
    incorrect_count     = 0
    parse_failed_count  = 0
    api_error_count     = 0

    with open(args.output, newline="", encoding="utf-8") as results_file:
        results_reader = csv.DictReader(results_file)
        for result_row in results_reader:
            if result_row["classification"] == "API_ERROR":
                api_error_count += 1
            elif result_row["classification"] == "PARSE_FAILED":
                parse_failed_count += 1
            elif result_row["correct"] == "True":
                correct_count += 1
            else:
                incorrect_count += 1

    scoreable   = correct_count + incorrect_count
    accuracy    = (correct_count / scoreable * 100) if scoreable > 0 else 0

    print(f"\n── RESULTS SUMMARY ──────────────────────────────")
    print(f"  Model:        {args.model}")
    print(f"  Condition:    {condition}")
    print(f"  Cases run:    {cases_run}")
    print(f"  Correct:      {correct_count}")
    print(f"  Incorrect:    {incorrect_count}")
    print(f"  Accuracy:     {accuracy:.1f}%")
    if parse_failed_count:
        print(f"  Parse failed: {parse_failed_count} (manual review needed)")
    if api_error_count:
        print(f"  API errors:   {api_error_count} (rerun to retry)")
    print(f"─────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
