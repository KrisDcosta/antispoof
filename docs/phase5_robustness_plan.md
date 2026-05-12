# Phase 5 Robustness Plan

Phase 5 evaluates whether the accepted LCNN, frozen WavLM, and frozen
LCNN+WavLM fusion systems remain reliable under deterministic synthetic
consumer-audio distortions on the ASVspoof 2019 LA eval split.

The implementation is intentionally evaluation-only:

- no retraining
- no retuning fusion weights
- no fitting normalization on corrupted eval
- no changes to accepted Phase 1-4 results

The frozen fusion rule is:

```text
fused_score = 0.7 * z_lcnn + 0.3 * z_wavlm
```

The runner is:

```bash
make robustness-smoke
make robustness-eval
```

Outputs are written under:

```text
results/robustness/phase5_eval_corruptions/
results/robustness/metrics/phase5_eval_corruptions_summary.json
```

The full run folder contains raw score CSVs and plots for review, while the
compact metrics summary is the only result artifact intended for git after the
user verifies the run.
