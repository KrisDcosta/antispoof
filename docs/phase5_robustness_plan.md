# Phase 5 Robustness Plan and Accepted Result

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

## Accepted Result

Completed full eval robustness run:

```text
run_id: phase5_eval_corruptions
elapsed_seconds: 13824.6
completed conditions: clean, gain, clipping, resampling, additive noise
skipped condition: codec_optional
```

Evidence:

```text
results/robustness/metrics/phase5_eval_corruptions_summary.json
```

Key EER results:

| Condition | LCNN | WavLM | Fusion |
|---|---:|---:|---:|
| clean | 5.67% | 5.08% | 3.62% |
| gain -12 dB | 5.67% | 5.29% | 3.51% |
| gain -6 dB | 5.67% | 5.10% | 3.52% |
| gain +6 dB | 5.67% | 5.19% | 3.71% |
| clip 0.8 | 5.81% | 5.09% | 3.79% |
| clip 0.6 | 6.28% | 5.24% | 5.19% |
| clip 0.4 | 8.03% | 5.74% | 6.13% |
| resample 12 kHz | 7.46% | 5.13% | 6.12% |
| resample 8 kHz | 7.57% | 4.72% | 5.79% |
| noise 20 dB | 16.48% | 11.45% | 15.48% |
| noise 10 dB | 34.49% | 21.62% | 36.53% |
| noise 5 dB | 41.22% | 38.55% | 39.80% |

Interpretation:

- Fusion is the strongest clean-eval system and remains stable under gain.
- WavLM is the most robust under additive noise and several stronger channel
  shifts.
- Fusion's LCNN-heavy weighting improves clean benchmark performance but is not
  always the most robust choice under corrupted audio.
- The research conclusion should separate clean complementarity from robustness:
  fusion is best for clean unseen attacks; SSL representations are safer under
  noisy channel shift.
