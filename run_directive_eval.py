"""
run_directive_eval.py

Convenience wrapper that combines narrative generation and LLM evaluation
into a single command for directive conditions.

Runs generate_narratives.py to produce a directive narrative CSV, then
immediately passes it to run_eval.py for evaluation.

Usage:
    python run_directive_eval.py \\
        --directive "A registered telephone number is a strong indicator of financial stability." \\
        --directive-label telephone_amplify \\
        --model anthropic/claude-haiku-4-5 \\
        --limit 50

Arguments:
    --directive        Explicit cue directive to inject into the prompt opening
    --directive-label  Short label for the output filename, e.g. telephone_amplify
    --model            OpenRouter model string e.g. anthropic/claude-haiku-4-5
    --input            Path to decoded CSV (default: german_credit_decoded.csv)
    --limit            Number of cases to evaluate (default: all)
    --output           Path to results CSV (default: auto-generated)
"""

import argparse
import subprocess
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="Generate directive narratives and run LLM eval in one step."
    )
    parser.add_argument("--directive",       required=True,  help="Explicit cue directive text")
    parser.add_argument("--directive-label", required=True,  help="Short label for output filename")
    parser.add_argument("--model",           required=True,  help="OpenRouter model string")
    parser.add_argument("--input",           default="data/german_credit_decoded.csv",
                                             help="Path to decoded CSV (default: data/german_credit_decoded.csv)")
    parser.add_argument("--limit",           type=int, default=None, help="Number of cases to evaluate")
    parser.add_argument("--output",          default=None,   help="Path to results CSV (default: auto-generated)")
    args = parser.parse_args()

    narrative_csv = f"data/german_credit_narratives_directive_{args.directive_label}.csv"

    # ── STEP 1: GENERATE DIRECTIVE NARRATIVES ─────────────────────────────────

    print(f"── Step 1: Generating directive narratives ───────────────────────")
    print(f"  Directive : {args.directive}")
    print(f"  Label     : {args.directive_label}")
    print(f"  Output    : {narrative_csv}")
    print()

    gen_cmd = [
        sys.executable, "generate_narratives.py",
        "--input",           args.input,
        "--directive",       args.directive,
        "--directive-label", args.directive_label,
    ]

    result = subprocess.run(gen_cmd)
    if result.returncode != 0:
        print("\nERROR: generate_narratives.py failed. Aborting.")
        sys.exit(1)

    if not os.path.exists(narrative_csv):
        print(f"\nERROR: Expected output file not found: {narrative_csv}")
        sys.exit(1)

    # ── STEP 2: RUN EVAL ──────────────────────────────────────────────────────

    print(f"\n── Step 2: Running eval ──────────────────────────────────────────")

    eval_cmd = [
        sys.executable, "run_eval.py",
        "--input", narrative_csv,
        "--model", args.model,
    ]
    if args.limit is not None:
        eval_cmd += ["--limit", str(args.limit)]
    if args.output is not None:
        eval_cmd += ["--output", args.output]

    result = subprocess.run(eval_cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
