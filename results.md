# Results Ledger

This file is the human-readable source of truth for reported project results.
Machine-readable evidence lives under `results/**/metrics/`, `results/**/summary/`,
and model-specific run folders. Any new result used in the README, app, report
assistant, or resume bullets should be added here with the run command and
evaluation setting.

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
| `lcnn_logmel_full_seed42_30ep` | Protocol-comparable | log-mel LCNN, no external pretraining | 0.75% | 5.67% | Current best project eval result | `results/neural/metrics/lcnn_logmel_full_seed42_30ep_summary.json` |
| `gmm_lfcc_64c_diag_std_300000frames_seed42` | Protocol-comparable | LFCC GMM-LLR | 0.06% | 10.25% | Strong classical baseline | `results/baseline/metrics/gmm_lfcc_64c_diag_std_300000frames_seed42_eval_metrics.json` |
| `aasist_lite_waveform_seed42_100ep` | Protocol-comparable | raw-waveform AASIST-lite, no external pretraining | 2.59% | 10.64% | Valid Phase 2 waveform graph-attention baseline; weaker eval generalization than LCNN | `results/neural/metrics/aasist_lite_waveform_seed42_100ep_summary.json` |
| `gmm_cqcc_64c_diag_std_300000frames_seed42` | Protocol-comparable | CQCC GMM-LLR | 11.15% | 11.59% | Valid classical run; simplified CQCC extractor | `results/baseline/metrics/gmm_cqcc_64c_diag_std_300000frames_seed42_eval_metrics.json` |
| `gmm_mfcc_64c_diag_std_300000frames_seed42` | Protocol-comparable | MFCC GMM-LLR | 9.77% | 16.41% | Valid classical run; weaker eval generalization | `results/baseline/metrics/gmm_mfcc_64c_diag_std_300000frames_seed42_eval_metrics.json` |

Summary artifacts:

- `results/neural/metrics/lcnn_logmel_full_seed42_30ep_summary.json`
- `results/neural/metrics/aasist_lite_waveform_seed42_100ep_summary.json`
- `results/baseline/summary/RESULTS.md`
- `results/baseline/summary/project_results.csv`
- `results/baseline/summary/plots/project_eer_by_split.png`
- `results/baseline/summary/plots/eval_eer_vs_standard_baselines.png`

## Interpretation

The Phase 1 log-mel LCNN is the current best project result:

- project LCNN eval EER: 5.67%
- published LFCC-GMM eval EER: 8.09%
- published CQCC-GMM eval EER: 9.57%
- improvement over published LFCC-GMM: 2.42 percentage points
- improvement over published CQCC-GMM: 3.90 percentage points

The model is trained from scratch with no external pretraining, so it remains in
the protocol-comparable track.

The Phase 2 AASIST-lite waveform model is also trained from scratch and remains
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
which is expected because eval contains unseen attack families. Future waveform,
SSL, and robustness models should be judged primarily by eval EER and per-attack
behavior. Accuracy is retained as a secondary diagnostic only; neural dev/eval
accuracy values use split-specific EER thresholds and should not be presented as
deployment-calibrated operating points.

## Planned Result Sections

The following sections should be filled as phases complete.

### Phase 1: PyTorch Spectrogram Countermeasures

Completed first full dev/eval GPU run in Colab on an A100.

Search decision:

- No broad grid search before the first valid neural baseline.
- First run uses the checked-in LCNN/log-mel config.
- Any later sweep must be targeted, dev-only, documented, and added to this
  ledger if used in project claims.

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

Primary failure modes are concentrated in A17, A18, A08, and A19. These should
drive the next robustness and explainability checks.

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

### Phase 2: Waveform and Graph-Attention Countermeasures

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
- It does not beat the Phase 1 LCNN or LFCC-GMM eval result.
- It remains valuable as the raw-waveform graph-attention baseline for later
  robustness and explainability comparisons.

### Phase 3: SSL Embedding Countermeasures

Pending.

Expected entries:

- frozen WavLM/wav2vec2 classifier
- adapter/LoRA or top-layer fine-tuned variant, if run
- explicit external-pretraining label
- dev/eval EER
- embedding cache notes

### Phase 4: Robustness Sweeps

Pending.

Expected entries:

- model
- corruption type
- corruption severity
- clean EER
- corrupted EER
- delta EER
- run artifact path

### Phase 5: Explainability

Pending.

Expected entries:

- GMM frame-level LLR traces
- CNN saliency/Grad-CAM examples
- AASIST attention examples, if reliable
- representative sample IDs
- interpretation caveats

### Phase 6: Agentic/RAG Assistant

Pending.

Expected entries:

- retrieval corpus version
- supported tools
- evaluation checks
- example report path
- hallucination/grounding safeguards

### Phase 7: Local Analyst App

Pending.

Expected entries:

- app command
- supported models
- supported visualizations
- screenshot paths
- limitations

## Change Log

| Date | Change |
|---|---|
| 2026-05-09 | Added Phase 2 AASIST-lite waveform full dev/eval result from Colab run. |
| 2026-05-08 | Added Phase 1 log-mel LCNN full dev/eval result from Colab A100 run. |
| 2026-05-07 | Created root results ledger and locked current classical GMM results. |
