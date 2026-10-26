# Supervisory-event audit

Audit code, anonymized aggregate results, and a fully synthetic example for
analysing supervisory events in industrial process records.

## Contents

- `scripts/`: event extraction interfaces, causal features, alert evaluation,
  transition episodes, nested-information comparisons, and sensitivity analyses.
- `artifacts/`: aggregate counts, metrics, intervals, calibration summaries,
  anonymized transition-pair summaries, and permutation summaries.
- `tests/`: synthetic timing, censoring, resampling, and aggregate consistency tests.
- `examples/synthetic_demo.py`: an invented event stream with known answers.
- `figures/`: three paper figures, reproducible as PDF and PNG from aggregate tables.
- `validation/`: public reproduction checks and redaction counts, without matched
  identifier values.

Plant records, individual event/episode rows, saved predictions, trained models,
private caches, original tag/grade/operator mappings, manuscript files, internal
reviews, and operational logs are excluded. Labels such as `grade 40` are
pseudonyms, not original grade codes. Aggregate cells can have small counts;
the release is anonymized but is not a differentially private dataset.

## Public reproduction

Use Python 3.11 with the versions in `requirements.txt`. The plotting font is
Liberation Serif, with STIXGeneral as fallback. In an environment containing
these dependencies, run from this directory:

```sh
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest tests -q
python examples/synthetic_demo.py
python scripts/paper1_figs/build_figures.py
python scripts/paper1_figs/verify_and_report.py
```

Alternatively, `python tools/validate_public.py` runs the public tests and figure
builder with an access guard that rejects record/model files, external artifact
directories, and user-installed research helpers. It records accessed aggregate
inputs in `validation/reproduction.json`. The PDF verifier checks 18 numerical
relationships, canvas width, minimum font size, and empty author metadata.
`.mplconfig/` is a disposable plotting cache and is excluded from distribution.

The figures use five CSV files: the manual-window summary, next-type metrics,
paired information increments, elapsed-matched co-occurrence controls, and
recent-target onset counts. Plotting recomputes probability differences and
recall from aggregate numerators/denominators; confidence intervals are the
published intervals. It does not refit models or resample confidential rows.

## Record-dependent stages

Analysis modules retain their numerical functions for inspection and synthetic
tests. Running a record-dependent command without private configuration prints
`SKIP: requires plant records` and exits without creating data. The historical
shell launchers likewise skip rather than start an experiment.

To adapt the analysis to another dataset, the data owner supplies an external
`PLANT_RECORDS_CONFIG` JSON file, a separate `AUDIT_WORKSPACE`, and, for ingestion
and masking, an `adapter_module`. The private configuration provides a `paths`
dictionary and DCS role descriptions. `validation/preparation.json` lists the
expected path keys and role-column keys without any original identifiers.
The adapter renames source channels to these public role keys, extracts the
record tables, and supplies the masking and grade-change interfaces. No private
configuration or source-to-role mapping belongs in this repository.

The public reproduction validates synthetic logic and figures from published
aggregates. Rebuilding the original event tables, re-estimating intervals, and
refitting the original models requires plant records and the owner-provided
interfaces. Those operations are outside the aggregate-only reproduction.

## Interpretation

The tables distinguish chronological evaluation from grouped evaluation and
decision weighting from episode weighting. Positive accuracy differences favour
the first model; negative log-loss differences favour the first model. The
elapsed-matched control preserves signed time relative to a grade transition;
it is distinct from shifts restricted to transition regions. Results quantify
conditional predictability and co-occurrence, not causal attribution.

## License

MIT; see `LICENSE`. Attribution is temporarily **Anonymous authors**.
