# Corrected German Credit (Study 2) analysis — convenience targets.
# Override the interpreter with `make validate PY=.venv/bin/python`.
PY ?= python

.PHONY: alignment protected validate corrected

# 14-cell alignment table + correlations + results CSV (analysis/corrected/).
alignment:
	$(PY) -m calm_corrected.run_alignment --repo .

# Corrected scatter + protected-attribute usage figure & CSV.
protected:
	$(PY) -m calm_corrected.run_protected_and_figures --repo .

# Regression guard: reproduces the published anchors, exits nonzero on mismatch.
validate:
	$(PY) -m calm_corrected.validate --repo .

# Everything.
corrected: alignment protected validate
