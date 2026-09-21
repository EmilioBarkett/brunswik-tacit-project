# Cue Utilization and Tacit Knowledge in LLM Decision-Making

Do LLMs weight information the way a decision-maker does, or do they only reach similar decisions by a different route? This project applies the **Brunswik Lens Model** to LLMs acting as credit officers, comparing the cues a model actually relies on with the cues a normative model says matter.

[![arXiv](https://img.shields.io/badge/arXiv-2605.25256-b31b1b.svg)](https://arxiv.org/abs/2605.25256)

Companion code and data for [*Whose Alignment? Comparing LLM Process Alignment Across Diverse Organizational Decision Contexts*](https://arxiv.org/abs/2605.25256) (Weller & Barkett, 2026), accepted to the **ICML 2026 Pluralistic Alignment Workshop**.

Part of a [SPAR](https://sparai.org/) research project. The paper covers two organizational settings. This repository holds **Study 2 (German Credit)**. Study 1, on European Court of Human Rights (ECHR) Article 6 decisions, is in a separate project.

## Research question

When an LLM makes a credit decision, how closely do its implicit cue weights match a normative policy, and can telling the model the policy (or feeding back its own biases) change that?

## Method

- **Data.** The Statlog [German Credit dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) (Hofmann, 1994): 1,000 loan applications described by 20 attributes, labeled Good or Bad credit risk.
- **Normative model.** A regularized logistic regression on the ground-truth labels gives normative cue weights, the "ecological validity" side of the lens model.
- **LLM judge.** Each application is rendered as a natural-language narrative. An LLM reasons through the decision (via OpenRouter), then a second scoring call (Claude Haiku) extracts the cue weights implied by that reasoning.
- **Alignment.** The LLM-implied weights are compared with the normative or organizational weights, mainly by cosine similarity of the coefficient vectors.
- **Conditions.** Narratives are generated under several conditions:
  - `bare`: the application only
  - `scenario`: the application in a scenario frame
  - `court_extern`: the model is told the normative cue rankings as organizational policy
  - `introspective`: the model is told which cues it over- and under-weighted, and the base rate
  - `directive`: an explicit instruction about one cue is injected (for example, telephone ownership)

## Repository layout

```
brunswik-tacit-project/
├── data/                               # German Credit data, decoded CSV, narratives per condition, normative weights
├── decode_german_credit.py             # Decode raw german.data into readable labels
├── normative_weights.py                # Fit the normative logistic regression
├── generate_narratives.py              # Bare, scenario, and directive narratives
├── generate_court_extern_narratives.py # Organization-externalized condition
├── generate_introspective_narratives.py# Introspective condition (per model)
├── run_eval.py                         # Run an LLM over a narrative CSV and extract cue weights
├── run_directive_eval.py               # Generate and evaluate a directive condition in one step
├── analyze_weights.py                  # Legacy alignment analysis (see note below)
├── calm_corrected/                     # Corrected alignment analysis (see note below)
├── plot_*.py                           # Figures
├── results/                            # Raw LLM outputs (main runs and pilots)
├── analysis/                           # Per-run analysis reports; corrected/ has the corrected outputs
├── report/                             # SPAR midterm report; icml2026/ has the paper source and figures
├── main_analysis_balanced_mistral-large.ipynb   # Cue-utilization alignment notebook (court vs. LLM setups)
├── llm_eval_list_v2.csv                # Registry of 40 candidate models
├── study_progress.csv                  # Progress tracker per model and condition
├── Makefile, requirements.txt          # Targets and dependencies for the corrected analysis
└── .env.example                        # Template for API keys (Anthropic, OpenRouter)
```

### Note on the analysis code

`calm_corrected/` is the current implementation of the Study 2 alignment analysis. It regresses LLM decisions and the organization's historical decisions on the actual cue values in a shared feature space. `analyze_weights.py` is the earlier implementation. It is kept only to reproduce previously reported numbers and is superseded by `calm_corrected/`. `make validate` checks the corrected pipeline against those earlier anchors.

## Project team

- **Niklas Weller**, University of St. Gallen
- **Emilio Barkett**, Columbia University

## Citation

```bibtex
@misc{weller2026whose,
  title         = {Whose Alignment? Comparing LLM Process Alignment Across Diverse Organizational Decision Contexts},
  author        = {Weller, Niklas and Barkett, Emilio},
  year          = {2026},
  eprint        = {2605.25256},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  note          = {Accepted to the ICML 2026 Pluralistic Alignment Workshop}
}
```

## References

- Brunswik, E. (1956). *Perception and the Representative Design of Psychological Experiments* (2nd ed.). University of California Press.
- Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository.
