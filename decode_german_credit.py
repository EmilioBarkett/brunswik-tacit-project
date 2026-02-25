"""
decode_german_credit.py
=======================
Decodes the German Credit Dataset (german.data) from encoded categorical
values into human-readable labels, outputting a clean CSV file.

Usage:
    python decode_german_credit.py

Input:  german.data        (space-separated, 21 values per row)
Output: german_credit_decoded.csv

=============================================================================
DECODING DICTIONARIES — verify these against the key (german.doc) before use
=============================================================================

Attribute 1 — Status of existing checking account
    A11 → "< 0 DM"
    A12 → "0 to < 200 DM"
    A13 → ">= 200 DM / salary assignment >= 1 year"
    A14 → "no checking account"

Attribute 3 — Credit history
    A30 → "no credits taken / all credits paid back duly"
    A31 → "all credits at this bank paid back duly"
    A32 → "existing credits paid back duly till now"
    A33 → "delay in paying off in the past"
    A34 → "critical account / other credits existing (not at this bank)"

Attribute 4 — Purpose
    A40  → "car (new)"
    A41  → "car (used)"
    A42  → "furniture/equipment"
    A43  → "radio/television"
    A44  → "domestic appliances"
    A45  → "repairs"
    A46  → "education"
    A47  → "vacation"
    A48  → "retraining"
    A49  → "business"
    A410 → "others"

Attribute 6 — Savings account/bonds
    A61 → "< 100 DM"
    A62 → "100 to < 500 DM"
    A63 → "500 to < 1000 DM"
    A64 → ">= 1000 DM"
    A65 → "unknown / no savings account"

Attribute 7 — Present employment since
    A71 → "unemployed"
    A72 → "< 1 year"
    A73 → "1 to < 4 years"
    A74 → "4 to < 7 years"
    A75 → ">= 7 years"

Attribute 9 — Personal status and sex
    A91 → "male: divorced/separated"
    A92 → "female: divorced/separated/married"
    A93 → "male: single"
    A94 → "male: married/widowed"
    A95 → "female: single"

Attribute 10 — Other debtors / guarantors
    A101 → "none"
    A102 → "co-applicant"
    A103 → "guarantor"

Attribute 12 — Property
    A121 → "real estate"
    A122 → "building society savings agreement / life insurance"
    A123 → "car or other, not in savings attribute"
    A124 → "unknown / no property"

Attribute 14 — Other installment plans
    A141 → "bank"
    A142 → "stores"
    A143 → "none"

Attribute 15 — Housing
    A151 → "rent"
    A152 → "own"
    A153 → "for free"

Attribute 17 — Job
    A171 → "unemployed / unskilled - non-resident"
    A172 → "unskilled - resident"
    A173 → "skilled employee / official"
    A174 → "management / self-employed / highly qualified employee / officer"

Attribute 19 — Telephone
    A191 → "none"
    A192 → "yes, registered under customer's name"

Attribute 20 — Foreign worker
    A201 → "yes"
    A202 → "no"

Attribute 21 — Credit risk (outcome)
    1 → "Good"
    2 → "Bad"
=============================================================================
"""

import csv

# ---------------------------------------------------------------------------
# INPUT / OUTPUT FILE PATHS — adjust if needed
# ---------------------------------------------------------------------------
INPUT_FILE  = "german.data"
OUTPUT_FILE = "german_credit_decoded.csv"

# ---------------------------------------------------------------------------
# DECODING DICTIONARIES (one per categorical attribute)
# ---------------------------------------------------------------------------

# Attribute 1: Status of existing checking account
checking_account_status = {
    "A11": "< 0 DM",
    "A12": "0 to < 200 DM",
    "A13": ">= 200 DM / salary assignment >= 1 year",
    "A14": "no checking account",
}

# Attribute 3: Credit history
credit_history = {
    "A30": "no credits taken / all credits paid back duly",
    "A31": "all credits at this bank paid back duly",
    "A32": "existing credits paid back duly till now",
    "A33": "delay in paying off in the past",
    "A34": "critical account / other credits existing (not at this bank)",
}

# Attribute 4: Purpose
purpose = {
    "A40":  "car (new)",
    "A41":  "car (used)",
    "A42":  "furniture/equipment",
    "A43":  "radio/television",
    "A44":  "domestic appliances",
    "A45":  "repairs",
    "A46":  "education",
    "A47":  "vacation",
    "A48":  "retraining",
    "A49":  "business",
    "A410": "others",
}

# Attribute 6: Savings account / bonds
savings_account = {
    "A61": "< 100 DM",
    "A62": "100 to < 500 DM",
    "A63": "500 to < 1000 DM",
    "A64": ">= 1000 DM",
    "A65": "unknown / no savings account",
}

# Attribute 7: Present employment since
employment_since = {
    "A71": "unemployed",
    "A72": "< 1 year",
    "A73": "1 to < 4 years",
    "A74": "4 to < 7 years",
    "A75": ">= 7 years",
}

# Attribute 9: Personal status and sex
personal_status_sex = {
    "A91": "male: divorced/separated",
    "A92": "female: divorced/separated/married",
    "A93": "male: single",
    "A94": "male: married/widowed",
    "A95": "female: single",
}

# Attribute 10: Other debtors / guarantors
other_debtors = {
    "A101": "none",
    "A102": "co-applicant",
    "A103": "guarantor",
}

# Attribute 12: Property
property_type = {
    "A121": "real estate",
    "A122": "building society savings agreement / life insurance",
    "A123": "car or other, not in savings attribute",
    "A124": "unknown / no property",
}

# Attribute 14: Other installment plans
other_installment_plans = {
    "A141": "bank",
    "A142": "stores",
    "A143": "none",
}

# Attribute 15: Housing
housing = {
    "A151": "rent",
    "A152": "own",
    "A153": "for free",
}

# Attribute 17: Job
job = {
    "A171": "unemployed / unskilled - non-resident",
    "A172": "unskilled - resident",
    "A173": "skilled employee / official",
    "A174": "management / self-employed / highly qualified employee / officer",
}

# Attribute 19: Telephone
telephone = {
    "A191": "none",
    "A192": "yes, registered under customer's name",
}

# Attribute 20: Foreign worker
foreign_worker = {
    "A201": "yes",
    "A202": "no",
}

# Attribute 21: Credit risk outcome
credit_risk_outcome = {
    "1": "Good",
    "2": "Bad",
}

# ---------------------------------------------------------------------------
# COLUMN DEFINITIONS
# Each entry is (snake_case_name, decoder_dict_or_None)
# None means the column is numerical and left as-is.
# ---------------------------------------------------------------------------
COLUMNS = [
    ("checking_account_status",   checking_account_status),   # col 1
    ("duration_months",           None),                       # col 2 — numerical
    ("credit_history",            credit_history),             # col 3
    ("purpose",                   purpose),                    # col 4
    ("credit_amount",             None),                       # col 5 — numerical
    ("savings_account_bonds",     savings_account),            # col 6
    ("employment_since",          employment_since),           # col 7
    ("installment_rate_pct",      None),                       # col 8 — numerical
    ("personal_status_sex",       personal_status_sex),        # col 9
    ("other_debtors_guarantors",  other_debtors),              # col 10
    ("present_residence_since",   None),                       # col 11 — numerical
    ("property",                  property_type),              # col 12
    ("age_years",                 None),                       # col 13 — numerical
    ("other_installment_plans",   other_installment_plans),    # col 14
    ("housing",                   housing),                    # col 15
    ("num_existing_credits",      None),                       # col 16 — numerical
    ("job",                       job),                        # col 17
    ("num_dependents",            None),                       # col 18 — numerical
    ("telephone",                 telephone),                  # col 19
    ("foreign_worker",            foreign_worker),             # col 20
    ("credit_risk",               credit_risk_outcome),        # col 21
]

# ---------------------------------------------------------------------------
# MAIN DECODING LOGIC
# ---------------------------------------------------------------------------
def decode_row(raw_values):
    """Decode a single row of 21 space-separated values into a dict."""
    decoded = {}
    for i, (col_name, decoder) in enumerate(COLUMNS):
        raw = raw_values[i]
        if decoder is None:
            # Numerical — keep as-is (strip whitespace just in case)
            decoded[col_name] = raw.strip()
        else:
            code = raw.strip()
            if code in decoder:
                decoded[col_name] = decoder[code]
            else:
                # Unknown code: preserve original and flag it
                decoded[col_name] = f"UNKNOWN({code})"
    return decoded


def main():
    header = [col_name for col_name, _ in COLUMNS]
    rows_processed = 0

    with open(INPUT_FILE, "r", newline="", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

        writer = csv.DictWriter(outfile, fieldnames=header)
        writer.writeheader()

        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue  # skip blank lines

            values = line.split()
            if len(values) != 21:
                print(f"  WARNING: line {line_num} has {len(values)} fields (expected 21) — skipped.")
                continue

            decoded_row = decode_row(values)
            writer.writerow(decoded_row)
            rows_processed += 1

    print(f"\nDone! {rows_processed} rows decoded and written to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
