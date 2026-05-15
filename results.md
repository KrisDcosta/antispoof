# Results Ledger

This file is the human-readable source of truth for reported project results.
Machine-readable evidence lives under `results/**/metrics/`, `results/**/summary/`,
and model-specific run folders. Result claims used in the README, reports, or
resume bullets should be traceable to this ledger.

## Reporting Rules

- Headline metric: eval EER.
- Dev EER is diagnostic, not the main claim.
- Report whether a run is protocol-comparable or uses external pretrained
  components.
- Keep clean ASVspoof eval separate from robustness/corrupted eval.
- Do not report accuracy as the headline metric.
- Do not claim SOTA unless the protocol, data, restrictions, and comparison
  setup match the cited work.

## Dataset

Dataset: ASVspoof 2019 Logical Access

| Split | Total | Bonafide | Spoof | Attack IDs |
|---|---:|---:|---:|---|
| Train | 25,380 | 2,580 | 22,800 | A01-A06 |
| Dev | 24,844 | 2,548 | 22,296 | A01-A06 |
| Eval | 71,237 | 7,355 | 63,882 | A07-A19 |

Eval attacks A07-A19 are unseen during training, so eval EER is the primary
generalization measurement.

## Published Reference Systems

| Reference System | Dev EER | Eval EER | Comparison Track | Source |
|---|---:|---:|---|---|
| Official LFCC-GMM | 2.71% | 8.09% | Protocol-comparable classical reference | ASVspoof 2019 challenge paper |
| Official CQCC-GMM | 0.43% | 9.57% | Protocol-comparable classical reference | ASVspoof 2019 challenge paper |

Reference metadata is stored in `references/standard_baselines.json`.

## Current Project Results

Configuration for current neural run:

- model: LCNN-style log-mel PyTorch countermeasure
- input: 4-second crop/pad waveform, 64-bin log-mel spectrogram
- optimizer: AdamW, learning rate 0.0003, weight decay 1e-6
- epochs: 30, best checkpoint selected by dev EER
- seed: 42
- external pretraining: false
- data: official ASVspoof 2019 LA train/dev/eval CM protocols

Neural run command:

```bash
python scripts/train_neural.py --config configs/neural_lcnn.json
```

Configuration for current classical runs:

- classifier: dual-GMM log-likelihood ratio
- covariance: diagonal
- GMM components: 64
- standardization: enabled
- max training frames per class: 300,000
- seed: 42
- data: official ASVspoof 2019 LA train/dev/eval CM protocols

Run family:

```bash
python scripts/train_eval.py \
  --data data/LA \
  --ablation \
  --splits dev eval \
  --cache-dir results/feature_cache \
  --output results/baseline
```

| Run ID | Track | Feature / Model | Dev EER | Eval EER | Status | Evidence |
|---|---|---|---:|---:|---|---|
| `lcnn_wavlm_score_fusion_seed42` | External-pretrained/applied | dev-normalized LCNN + WavLM weighted score fusion | 0.55% | 3.62% | Best numeric project result; selected by dev EER | `results/fusion/metrics/lcnn_wavlm_score_fusion_seed42_summary.json` |
| `lcnn_logmel_full_seed42_30ep` | Protocol-comparable | log-mel LCNN, no external pretraining | 0.75% | 5.67% | Best protocol-comparable project eval result | `results/neural/metrics/lcnn_logmel_full_seed42_30ep_summary.json` |
| `ssl_wavlm_pooled_full_seed42_50ep` | External-pretrained/applied | frozen WavLM-base-plus mean+std MLP | 3.02% | 5.08% | Best individual applied model; not protocol-comparable because WavLM uses external pretraining | `results/neural/metrics/ssl_wavlm_pooled_full_seed42_50ep_summary.json` |
| `gmm_lfcc_64c_diag_std_300000frames_seed42` | Protocol-comparable | LFCC GMM-LLR | 0.06% | 10.25% | Strong classical baseline | `results/baseline/metrics/gmm_lfcc_64c_diag_std_300000frames_seed42_eval_metrics.json` |
| `aasist_lite_waveform_seed42_100ep` | Protocol-comparable | raw-waveform AASIST-lite, no external pretraining | 2.59% | 10.64% | Valid waveform graph-attention baseline; weaker eval generalization than LCNN | `results/neural/metrics/aasist_lite_waveform_seed42_100ep_summary.json` |
| `gmm_cqcc_64c_diag_std_300000frames_seed42` | Protocol-comparable | CQCC GMM-LLR | 11.15% | 11.59% | Valid classical run; simplified CQCC extractor | `results/baseline/metrics/gmm_cqcc_64c_diag_std_300000frames_seed42_eval_metrics.json` |
| `gmm_mfcc_64c_diag_std_300000frames_seed42` | Protocol-comparable | MFCC GMM-LLR | 9.77% | 16.41% | Valid classical run; weaker eval generalization | `results/baseline/metrics/gmm_mfcc_64c_diag_std_300000frames_seed42_eval_metrics.json` |

Summary artifacts:

- `results/neural/metrics/lcnn_logmel_full_seed42_30ep_summary.json`
- `results/neural/metrics/aasist_lite_waveform_seed42_100ep_summary.json`
- `results/neural/metrics/ssl_wavlm_pooled_full_seed42_50ep_summary.json`
- `results/fusion/metrics/lcnn_wavlm_score_fusion_seed42_summary.json`
- `results/robustness/metrics/phase5_eval_corruptions_summary.json`
- `results/baseline/summary/RESULTS.md`
- `results/baseline/summary/project_results.csv`
- `results/baseline/summary/plots/project_eer_by_split.png`
- `results/baseline/summary/plots/eval_eer_vs_standard_baselines.png`

## Interpretation

The log-mel LCNN is the best protocol-comparable project result:

- project LCNN eval EER: 5.67%
- published LFCC-GMM eval EER: 8.09%
- published CQCC-GMM eval EER: 9.57%
- improvement over published LFCC-GMM: 2.42 percentage points
- improvement over published CQCC-GMM: 3.90 percentage points

The model is trained from scratch with no external pretraining, so it remains in
the protocol-comparable track.

The frozen WavLM pooled embedding classifier is the best individual applied
model:

- project WavLM pooled MLP eval EER: 5.08%
- project LCNN eval EER: 5.67%
- track: external-pretrained/applied

This result must be reported separately from the protocol-comparable ASVspoof
challenge-style systems because `microsoft/wavlm-base-plus` was pretrained on
external speech data. It is useful as an applied audio ML comparison showing
that frozen SSL speech representations contain spoof-relevant information.

LCNN+WavLM score fusion is the best numeric project result:

- project fusion eval EER: 3.62%
- project WavLM pooled MLP eval EER: 5.08%
- project LCNN eval EER: 5.67%
- selected rule: `0.7 * z_lcnn + 0.3 * z_wavlm`

This fusion result is external-pretrained/applied because it includes WavLM
scores. It supports the research hypothesis that the scratch-trained
spectrogram model and external-pretrained SSL model make complementary errors.

The robustness sweep tests whether that complementarity survives deterministic
audio corruptions. Fusion remains strong under gain shifts and moderate
clipping/resampling, but additive noise reverses the ranking: WavLM is the most
robust under noise, while fusion inherits enough LCNN sensitivity to degrade
more sharply.

The AASIST-lite waveform model is also trained from scratch and remains
protocol-comparable, but it is not the strongest eval result:

- project AASIST-lite waveform eval EER: 10.64%
- project LCNN eval EER: 5.67%
- project LFCC-GMM eval EER: 10.25%

This result is still useful because it provides the project with a raw-waveform
graph-attention baseline and exposes a different failure profile. Most eval
errors are concentrated in A17, A18, and A19, suggesting that the compact
waveform graph model is more brittle on the hardest unseen attack families than
the log-mel LCNN.

The LFCC GMM-LLR system is still a strong classical baseline and is close to
the published LFCC-GMM eval reference, but does not match it:

- project LFCC-GMM eval EER: 10.25%
- published LFCC-GMM eval EER: 8.09%
- gap: 2.16 percentage points

The dev result is much stronger than eval for both classical and neural models,
which is expected because eval contains unseen attack families. Accuracy is
retained as a secondary diagnostic only; neural dev/eval accuracy values use
split-specific EER thresholds and should not be presented as deployment-calibrated
operating points.

## Completed Run Details

### Log-Mel LCNN Countermeasure

Completed first full dev/eval GPU run in Colab on an A100.

Search decision:

- No broad grid search before the first valid neural baseline.
- First run uses the checked-in LCNN/log-mel config.
- The accepted project result is the full run below; no undocumented search
  trials are used in the project claims.

Full-run validation:

| Run ID | Track | Feature / Model | Params | Train Time | Dev EER | Eval EER | Evidence |
|---|---|---|---:|---:|---:|---:|---|
| `lcnn_logmel_full_seed42_30ep` | Protocol-comparable | log-mel LCNN | 665,153 | 553.1s | 0.75% | 5.67% | `results/neural/metrics/lcnn_logmel_full_seed42_30ep_summary.json` |

Eval per-attack EER:

| Attack | EER |
|---|---:|
| A07 | 0.50% |
| A08 | 4.42% |
| A09 | 0.65% |
| A10 | 0.88% |
| A11 | 0.57% |
| A12 | 0.83% |
| A13 | 0.94% |
| A14 | 1.04% |
| A15 | 0.68% |
| A16 | 0.83% |
| A17 | 20.39% |
| A18 | 8.74% |
| A19 | 3.55% |

Primary failure modes are concentrated in A17, A18, A08, and A19.

Smoke validation:

| Run ID | Track | Feature / Model | Dev EER | Status | Evidence |
|---|---|---|---:|---|---|
| `lcnn_logmel_smoke_seed42_1ep` | Protocol-comparable | log-mel LCNN smoke | 53.12% | Pipeline validation only; not a meaningful model result | `results/neural/lcnn_logmel_smoke_seed42_1ep/metrics.json` |

Smoke command:

```bash
make neural-smoke
```

Smoke interpretation:

- The run used 64 training utterances, 64 dev utterances, and 1 epoch.
- The result is not reported as a model benchmark.
- The run validates dataloading, log-mel extraction, training, scoring,
  metrics, plots, checkpointing, and model-card generation.

### AASIST-Lite Waveform Countermeasure

Completed first full dev/eval GPU run in Colab.

Run command:

```bash
python scripts/train_neural.py --config configs/neural_aasist_lite.json
```

Full-run validation:

| Run ID | Track | Feature / Model | Params | Train Time | Dev EER | Eval EER | Evidence |
|---|---|---|---:|---:|---:|---:|---|
| `aasist_lite_waveform_seed42_100ep` | Protocol-comparable | raw-waveform AASIST-lite | 137,828 | 4298.8s | 2.59% | 10.64% | `results/neural/metrics/aasist_lite_waveform_seed42_100ep_summary.json` |

Eval per-attack EER:

| Attack | EER |
|---|---:|
| A07 | 0.31% |
| A08 | 4.27% |
| A09 | 0.31% |
| A10 | 0.45% |
| A11 | 0.45% |
| A12 | 0.30% |
| A13 | 0.29% |
| A14 | 0.31% |
| A15 | 0.39% |
| A16 | 0.88% |
| A17 | 34.40% |
| A18 | 25.68% |
| A19 | 11.58% |

Interpretation:

- The model is effective on many eval attacks but weak on A17, A18, and A19.
- It does not beat the LCNN or LFCC-GMM eval result.
- It provides a raw-waveform graph-attention baseline inside the same
  evaluation harness as the classical and LCNN systems.

### Frozen WavLM SSL Embedding Countermeasure

Completed full dev/eval GPU run in Colab.

Run command:

```bash
python scripts/cache_ssl_embeddings.py --config configs/neural_ssl_wavlm_frozen.json
python scripts/train_ssl_head.py --config configs/neural_ssl_wavlm_frozen.json
```

Full-run validation:

| Run ID | Track | Feature / Model | Params | Train Time | Dev EER | Eval EER | Evidence |
|---|---|---|---:|---:|---:|---:|---|
| `ssl_wavlm_pooled_full_seed42_50ep` | External-pretrained/applied | frozen WavLM-base-plus mean+std MLP | 393,986 | 58.8s | 3.02% | 5.08% | `results/neural/metrics/ssl_wavlm_pooled_full_seed42_50ep_summary.json` |

Eval per-attack EER:

| Attack | EER |
|---|---:|
| A07 | 0.37% |
| A08 | 2.73% |
| A09 | 0.15% |
| A10 | 1.11% |
| A11 | 0.69% |
| A12 | 0.22% |
| A13 | 0.06% |
| A14 | 0.86% |
| A15 | 0.63% |
| A16 | 0.60% |
| A17 | 6.57% |
| A18 | 8.02% |
| A19 | 19.82% |

Interpretation:

- The WavLM pooled MLP is the best individual applied model before fusion.
- It is not protocol-comparable to ASVspoof challenge systems because the
  frozen encoder uses external pretraining.
- It sharply improves A17 and A18 compared with the scratch-trained LCNN, but
  A19 remains the largest eval weakness.

### LCNN + WavLM Score Fusion

Completed full dev/eval score-fusion run locally.

Run command:

```bash
make score-fusion
```

The fusion runner aligns score CSVs by `file_id`, verifies labels and attack
IDs, normalizes scores with dev-only statistics, evaluates mean / weighted mean
/ logistic-regression fusion, selects by dev EER, then applies the frozen rule
to eval.

Full-run validation:

| Run ID | Track | Sources | Selected Rule | Dev EER | Eval EER | Evidence |
|---|---|---|---|---:|---:|---|
| `lcnn_wavlm_score_fusion_seed42` | External-pretrained/applied | LCNN + WavLM | weighted mean, `alpha=0.7` toward LCNN | 0.55% | 3.62% | `results/fusion/metrics/lcnn_wavlm_score_fusion_seed42_summary.json` |

Fusion methods:

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

Interpretation:

- Fusion beats both individual systems on eval EER.
- Fusion keeps WavLM's A17 improvement while recovering LCNN-like strength on
  A19.
- The selected `alpha=0.7` means the scratch-trained LCNN receives more weight,
  but WavLM contributes enough complementary information to reduce overall EER.

### Robustness Evaluation

Completed full eval robustness sweep in Colab.

Run command:

```bash
python scripts/run_robustness_eval.py --config configs/robustness_eval.json
```

The robustness runner evaluates the frozen LCNN, frozen WavLM, and frozen
score-fusion rule on deterministic corruptions. It does not retrain models, refit
normalization, or retune fusion weights.

Runtime:

| Run ID | Split | Conditions | Elapsed Time | Evidence |
|---|---|---:|---:|---|
| `phase5_eval_corruptions` | eval | 12 completed, 1 codec condition skipped | 13,824.6s | `results/robustness/metrics/phase5_eval_corruptions_summary.json` |

Selected EER by condition:

| Condition | LCNN EER | WavLM EER | Fusion EER | Interpretation |
|---|---:|---:|---:|---|
| clean | 5.67% | 5.08% | 3.62% | fusion is best on clean eval |
| gain -12 dB | 5.67% | 5.29% | 3.51% | gain does not harm fusion |
| gain -6 dB | 5.67% | 5.10% | 3.52% | gain does not harm fusion |
| gain +6 dB | 5.67% | 5.19% | 3.71% | small fusion degradation |
| clip 0.8 | 5.81% | 5.09% | 3.79% | mild clipping has small impact |
| clip 0.6 | 6.28% | 5.24% | 5.19% | fusion advantage narrows |
| clip 0.4 | 8.03% | 5.74% | 6.13% | WavLM is strongest under heavy clipping |
| resample 12 kHz | 7.46% | 5.13% | 6.12% | WavLM is most stable |
| resample 8 kHz | 7.57% | 4.72% | 5.79% | WavLM improves under 8 kHz resampling |
| noise 20 dB | 16.48% | 11.45% | 15.48% | WavLM is most noise-robust |
| noise 10 dB | 34.49% | 21.62% | 36.53% | additive noise is the main failure mode |
| noise 5 dB | 41.22% | 38.55% | 39.80% | all systems degrade severely |

Skipped condition:

```text
codec_optional: codec corruption is optional and not enabled
```

Interpretation:

- Fusion remains the best system for clean audio, gain shifts, mild clipping,
  and clean benchmark-style conditions.
- WavLM is consistently more robust than LCNN and fusion under additive noise.
- Heavy clipping and resampling reduce the fusion advantage because the
  LCNN-weighted fusion rule inherits spectrogram-model sensitivity.
- The final interpretation separates clean benchmark performance from
  corruption behavior: fusion improves clean unseen-attack performance, while
  SSL representations are more stable under several channel distortions.
