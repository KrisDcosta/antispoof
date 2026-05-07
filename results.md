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
| `gmm_lfcc_64c_diag_std_300000frames_seed42` | Protocol-comparable | LFCC GMM-LLR | 0.06% | 10.25% | Current best project eval result | `results/baseline/metrics/gmm_lfcc_64c_diag_std_300000frames_seed42_eval_metrics.json` |
| `gmm_cqcc_64c_diag_std_300000frames_seed42` | Protocol-comparable | CQCC GMM-LLR | 11.15% | 11.59% | Valid classical run; simplified CQCC extractor | `results/baseline/metrics/gmm_cqcc_64c_diag_std_300000frames_seed42_eval_metrics.json` |
| `gmm_mfcc_64c_diag_std_300000frames_seed42` | Protocol-comparable | MFCC GMM-LLR | 9.77% | 16.41% | Valid classical run; weaker eval generalization | `results/baseline/metrics/gmm_mfcc_64c_diag_std_300000frames_seed42_eval_metrics.json` |

Summary artifacts:

- `results/baseline/summary/RESULTS.md`
- `results/baseline/summary/project_results.csv`
- `results/baseline/summary/plots/project_eer_by_split.png`
- `results/baseline/summary/plots/eval_eer_vs_standard_baselines.png`

## Interpretation

The LFCC GMM-LLR system is close to the published LFCC-GMM eval reference, but
does not match it:

- project LFCC-GMM eval EER: 10.25%
- published LFCC-GMM eval EER: 8.09%
- gap: 2.16 percentage points

The dev result is much stronger than eval, which is expected because eval
contains unseen attack families. Future neural, waveform, and SSL models should
be judged primarily by eval EER and per-attack behavior.

## Planned Result Sections

The following sections should be filled as phases complete.

### Phase 1: PyTorch Spectrogram Countermeasures

Implementation in progress. The local smoke run has passed; full dev/eval
training is pending Colab/GPU execution.

Search decision:

- No broad grid search before the first valid neural baseline.
- First run uses the checked-in LCNN/log-mel config.
- Any later sweep must be targeted, dev-only, documented, and added to this
  ledger if used in project claims.

Expected entries:

- log-mel CNN
- LFCC-LCNN or LCNN-style model
- dev/eval EER
- per-attack EER
- parameter count
- training command
- run artifact path

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

Pending.

Expected entries:

- AASIST-lite
- optional RawNet2-style model
- dev/eval EER
- per-attack EER
- parameter count
- training command
- run artifact path

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
| 2026-05-07 | Created root results ledger and locked current classical GMM results. |
