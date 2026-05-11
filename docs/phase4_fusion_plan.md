# Phase 4 Score Fusion Plan and Accepted Result

Phase 4 tests whether the strongest protocol-comparable model and the strongest
external-pretrained applied model make complementary errors. The phase uses
score-level fusion only; it does not retrain audio feature extractors or neural
front ends.

Accepted full result:

| Run ID | Selected Rule | Sources | Dev EER | Eval EER | Evidence |
|---|---|---|---:|---:|---|
| `lcnn_wavlm_score_fusion_seed42` | weighted mean, `alpha=0.7` toward LCNN | LCNN + WavLM | 0.55% | 3.62% | `results/fusion/metrics/lcnn_wavlm_score_fusion_seed42_summary.json` |

## Why Fusion

The LCNN and WavLM systems have different attack-family weaknesses. The LCNN is
trained from scratch and remains protocol-comparable; it is much stronger than
WavLM on A19. The frozen WavLM system uses external speech pretraining and is
much stronger than LCNN on A17. Score fusion tests whether those differences are
complementary rather than redundant.

Fusion is deliberately done at the score level because the individual systems
already produce calibrated evidence about each utterance. This keeps Phase 4
cheap, interpretable, and less prone to overfitting than training another audio
model.

## Method

Inputs:

```text
results/neural/lcnn_logmel_full_seed42_30ep/scores/{dev,eval}_scores.csv
results/neural/ssl_wavlm_pooled_full_seed42_50ep/scores/{dev,eval}_scores.csv
```

The runner:

1. aligns score files by `file_id`
2. verifies labels and attack IDs match across sources
3. fits z-score normalization using dev scores only
4. evaluates mean fusion, weighted mean fusion, and logistic-regression fusion
5. selects the rule with the lowest dev EER
6. applies the frozen selected rule to eval

The selected rule is:

```text
fused_score = 0.7 * z_lcnn + 0.3 * z_wavlm
```

## Results

| Method | Dev EER | Eval EER |
|---|---:|---:|
| mean | 0.56% | 3.79% |
| weighted mean | 0.55% | 3.62% |
| logistic regression | 0.55% | 3.81% |

Selected eval per-attack EER:

| Attack | EER |
|---|---:|
| A07 | 0.18% |
| A08 | 3.40% |
| A09 | 0.29% |
| A10 | 0.50% |
| A11 | 0.39% |
| A12 | 0.33% |
| A13 | 0.35% |
| A14 | 0.50% |
| A15 | 0.33% |
| A16 | 0.37% |
| A17 | 6.15% |
| A18 | 4.35% |
| A19 | 3.10% |

## Interpretation

The fusion result is research-useful because it beats both individual systems:

- LCNN eval EER: 5.67%
- frozen WavLM eval EER: 5.08%
- LCNN+WavLM fusion eval EER: 3.62%

The selected `alpha=0.7` shows that the LCNN carries more useful eval weight
overall, but WavLM still contributes complementary information. The largest
research signal is on the hard attack families: fusion keeps A19 close to the
LCNN behavior while retaining much of WavLM's gain on A17/A18.

## Guardrails

- Dev statistics are used for score normalization.
- Dev EER selects the fusion rule.
- Eval is used only after the rule is frozen.
- Raw score CSVs remain out of git.
- The result is not SOTA; it is a controlled complementarity result.
- Because WavLM is included, the fusion is an applied external-pretrained
  result, not protocol-comparable.
