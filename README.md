# Speech Anti-Spoofing on ASVspoof 2019 Logical Access

This project builds a reproducible speech anti-spoofing system for the
ASVspoof 2019 Logical Access task. It includes classical countermeasures based
on frame-level cepstral features and Gaussian mixture models, a PyTorch
LCNN-style neural countermeasure trained on log-mel spectrograms, and a
raw-waveform AASIST-lite graph-attention baseline. It also includes an
external-pretrained frozen WavLM embedding classifier as an applied comparison.

The result is an end-to-end audio ML workflow with dataset validation, EDA,
feature extraction, model training, scoring, EER evaluation, per-attack
diagnostics, and comparison against published ASVspoof reference systems.
The repository is organized as a finished experiment snapshot: reproducible
commands, compact result evidence, and documentation are committed, while the
large ASVspoof audio, raw score files, checkpoints, and embedding caches remain
outside git.

For a short project report, start with `docs/project_summary.md`. For rerunning
or auditing the results, start with `docs/reproducibility.md`.

Documentation:

- `results.md`: authoritative results ledger for reported numbers
- `docs/project_summary.md`: concise final project report and architecture map
- `docs/reproducibility.md`: setup, artifact, and rerun guide

## Method

The classical classifier is a dual-GMM log-likelihood-ratio system:

```text
score(utterance) = mean_t log p(x_t | GMM_bonafide) - mean_t log p(x_t | GMM_spoof)
```

Higher scores indicate bonafide speech. Equal Error Rate (EER) is computed from
the continuous utterance scores and is the primary metric because the dataset is
class-imbalanced.

Implemented frame-level front ends:

- LFCC with delta and delta-delta coefficients
- MFCC
- CQCC
- WCQCC, an exploratory weighted CQCC variant

The main configuration uses 64-component diagonal-covariance GMMs, standardized
frame features, a fixed random seed, and up to 300,000 sampled training frames
per class.

The neural baseline is an LCNN-style PyTorch model trained from scratch on
fixed-length log-mel spectrograms, with the best checkpoint selected by dev EER.
The waveform baseline is an AASIST-lite inspired model trained directly on
fixed-length 16 kHz waveform crops, also from scratch and without external
pretraining.
The applied SSL baseline freezes `microsoft/wavlm-base-plus`, pools
`last_hidden_state` with mean+std statistics, and trains a small MLP classifier
on cached embeddings. This WavLM result is external-pretrained and is reported
separately from protocol-comparable ASVspoof challenge systems.
The score-fusion system combines dev-normalized LCNN and WavLM scores with a
dev-selected weighted mean, testing whether the two model families make
complementary errors.
The robustness evaluation reuses the frozen LCNN, WavLM, and fusion systems
under deterministic gain, clipping, resampling, and additive-noise corruptions.

## Results

Full dev/eval runs were completed on the official ASVspoof 2019 LA protocol.
Eval is the main generalization split because attacks A07-A19 are unseen during
training.

| Method | Dev EER | Eval EER | Interpretation |
|---|---:|---:|---|
| log-mel LCNN, no external pretraining | 0.75% | 5.67% | Strongest protocol-comparable project result; trained from scratch on ASVspoof LA |
| LFCC GMM-LLR | 0.06% | 10.25% | Strong classical baseline; close to the published LFCC-GMM eval reference |
| AASIST-lite waveform, no external pretraining | 2.59% | 10.64% | Valid raw-waveform graph-attention baseline; weak on hardest unseen attacks |
| CQCC GMM-LLR | 11.15% | 11.59% | Protocol-correct, but uses a simplified Python CQCC extractor |
| MFCC GMM-LLR | 9.77% | 16.41% | Fits dev better than CQCC, generalizes worse on unseen eval attacks |

External-pretrained applied result:

| Method | Track | Dev EER | Eval EER | Interpretation |
|---|---|---:|---:|---|
| frozen WavLM-base-plus mean+std MLP | external-pretrained/applied | 3.02% | 5.08% | Best individual applied model, but not protocol-comparable because WavLM uses external pretraining |
| LCNN + WavLM score fusion | external-pretrained/applied | 0.55% | 3.62% | Best numeric project result; selected by dev EER with weighted mean fusion |

Published ASVspoof 2019 LA reference systems:

| Reference System | Dev EER | Eval EER | Source |
|---|---:|---:|---|
| Official LFCC-GMM | 2.71% | 8.09% | [ASVspoof 2019 challenge paper](https://arxiv.org/abs/1904.05441) |
| Official CQCC-GMM | 0.43% | 9.57% | [ASVspoof 2019 challenge paper](https://arxiv.org/abs/1904.05441) |

The LCNN improves over the published LFCC-GMM eval reference by 2.42 percentage
points and over the published CQCC-GMM eval reference by 3.90 percentage points.
Its largest eval weaknesses are A17, A18, A08, and A19, which define the
residual error profile of the strongest model.

The AASIST-lite waveform run is a valid protocol-comparable raw-waveform
graph-attention baseline, but it does not beat the LCNN or LFCC GMM eval
result. Its errors are concentrated in A17, A18, and A19, which makes it useful
for comparing waveform/graph-attention behavior against the spectrogram LCNN
within the same reproducible evaluation harness.

The frozen WavLM embedding classifier reaches 5.08% eval EER as the best
individual applied model, and LCNN+WavLM score fusion improves the numeric eval
EER to 3.62%. These results are external-pretrained/applied because they use
WavLM-derived scores. They are useful as evidence that pretrained speech
representations carry spoof-relevant information, while the LCNN remains the
strongest protocol-comparable model trained from scratch in this repository.

Robustness evaluation shows the fusion system is stable under gain changes and
moderate clipping/resampling, but additive noise is the dominant failure mode.
Under 10 dB additive noise, EER rises to 34.49% for LCNN, 21.62% for WavLM, and
36.53% for fusion, showing that WavLM is the most noise-robust model even though
fusion is strongest on clean eval.

Generated reports:

- `results.md`
- `results/neural/metrics/lcnn_logmel_full_seed42_30ep_summary.json`
- `results/neural/metrics/aasist_lite_waveform_seed42_100ep_summary.json`
- `results/neural/metrics/ssl_wavlm_pooled_full_seed42_50ep_summary.json`
- `results/fusion/metrics/lcnn_wavlm_score_fusion_seed42_summary.json`
- `results/robustness/metrics/phase5_eval_corruptions_summary.json`
- `results/baseline/summary/RESULTS.md`
- `results/baseline/summary/project_results.csv`
- `results/baseline/summary/plots/project_eer_by_split.png`
- `results/baseline/summary/plots/eval_eer_vs_standard_baselines.png`
- ROC, score distribution, and per-attack EER plots under
  `results/baseline/plots/`

## Dataset

Download ASVspoof 2019 LA from the official challenge page and place it at:

```text
data/LA/
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
├── ASVspoof2019_LA_eval/flac/
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

Local protocol counts:

| Split | Total | Bonafide | Spoof | Attack IDs |
|---|---:|---:|---:|---|
| Train | 25,380 | 2,580 | 22,800 | A01-A06 |
| Dev | 24,844 | 2,548 | 22,296 | A01-A06 |
| Eval | 71,237 | 7,355 | 63,882 | A07-A19 |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run tests:

```bash
make test
```

Install neural-model dependencies:

```bash
pip install -r requirements-neural.txt
```

## Reproduce The Project

The experiment settings are stored in:

```text
configs/asvspoof2019_gmm.json
```

Run the complete workflow:

```bash
make project
```

This runs EDA, trains/scores the configured LFCC/MFCC/CQCC systems on dev and
eval, and regenerates the final result tables and comparison plots.

For a fast pipeline check:

```bash
make smoke
```

The smoke run uses the same orchestration path with a small deterministic
class-balanced sample cap and writes to `results/smoke_project/`.

Equivalent direct command:

```bash
python scripts/run_project.py --config configs/asvspoof2019_gmm.json
```

Run the LCNN neural smoke test:

```bash
make neural-smoke
```

The neural smoke test validates the PyTorch log-mel LCNN pipeline locally. Its
EER is not a benchmark because it uses a tiny capped sample and one epoch.

Run the full LCNN neural baseline on GPU:

```bash
python scripts/train_neural.py --config configs/neural_lcnn.json
```

The recorded full run used a Colab A100 and produced
`lcnn_logmel_full_seed42_30ep`.

Run the AASIST-lite waveform smoke test:

```bash
make aasist-smoke
```

Run the full AASIST-lite waveform baseline on GPU:

```bash
python scripts/train_neural.py --config configs/neural_aasist_lite.json
```

Run the frozen WavLM applied SSL baseline on GPU:

```bash
python scripts/cache_ssl_embeddings.py --config configs/neural_ssl_wavlm_frozen.json
python scripts/train_ssl_head.py --config configs/neural_ssl_wavlm_frozen.json
```

Run LCNN+WavLM score fusion after restoring both systems' score CSVs:

```bash
make score-fusion
```

Run the robustness sweep after restoring eval audio and model artifacts:

```bash
make robustness-smoke
make robustness-eval
```

## Common Commands

Run EDA only:

```bash
python scripts/eda.py --data data/LA --output results/eda
```

Train and evaluate LFCC only:

```bash
python scripts/train_eval.py \
  --data data/LA \
  --feature lfcc \
  --splits dev eval \
  --cache-dir results/feature_cache \
  --output results/baseline
```

Run the LFCC/MFCC/CQCC comparison:

```bash
python scripts/train_eval.py \
  --data data/LA \
  --ablation \
  --splits dev eval \
  --cache-dir results/feature_cache \
  --output results/baseline
```

Regenerate final tables and comparison plots from saved metrics:

```bash
python scripts/summarize_results.py \
  --results results/baseline \
  --output results/baseline/summary
```

Regenerate per-attack analysis from a saved score file:

```bash
python scripts/attack_breakdown.py \
  --scores results/baseline/scores/<score_file>.csv \
  --output results/baseline/plots
```

## Outputs

Each full run writes:

```text
results/baseline/
├── metrics/
│   ├── *_metrics.json
│   └── ablation_summary.json
├── models/
│   └── *.joblib
├── plots/
│   ├── *_roc.png
│   ├── *_score_distribution.png
│   └── *_per_attack_eer.png
├── scores/
│   └── *_scores.csv
└── summary/
    ├── RESULTS.md
    ├── project_results.csv
    └── plots/
```

Score CSV columns include `file_id`, `label`, `system_id`, `score`, feature,
classifier, split, per-model log-likelihoods, and frame counts.

Feature caching is enabled with `--cache-dir`. The cache stores versioned frame
matrices keyed by feature type and audio metadata, so repeated experiments avoid
re-extracting the same utterances. Cache files and model artifacts are ignored by
git because they are large and reproducible.

## Technical Notes

- LFCC is the strongest implemented front end for this repository.
- The CQCC implementation uses `librosa.cqt` plus DCT. Exact reproduction of
  the official CQCC-GMM number would require matching the official CQCC feature
  extraction recipe.
- Accuracy is reported for completeness, but EER and per-attack EER are the
  meaningful metrics for this task.
- The archived RandomForest/mean-pooled implementation is preserved under
  `archive/original_rf_pipeline_2026-05-05/`. It is not used by the project
  workflow because mean pooling discards the frame-level distribution needed by
  GMM log-likelihood scoring.

## Project Structure

```text
SAP/
├── configs/
│   ├── asvspoof2019_gmm.json
│   ├── asvspoof2019_smoke.json
│   ├── neural_aasist_lite.json
│   ├── neural_aasist_lite_smoke.json
│   ├── neural_lcnn.json
│   ├── neural_lcnn_smoke.json
│   ├── neural_ssl_wavlm_frozen.json
│   ├── neural_ssl_wavlm_frozen_smoke.json
│   ├── robustness_eval.json
│   └── robustness_eval_smoke.json
├── docs/
│   ├── project_summary.md
│   └── reproducibility.md
├── results.md
├── src/
│   ├── dataset.py        # official ASVspoof protocol loader
│   ├── evaluate.py       # EER, ROC, confusion matrix utilities
│   ├── feature_cache.py  # versioned frame feature cache
│   ├── features.py       # LFCC/MFCC/CQCC/WCQCC extractors
│   ├── gmm_baseline.py   # dual-GMM training and LLR scoring
│   ├── neural/           # PyTorch datasets, LCNN, AASIST-lite, SSL heads
│   └── reporting.py      # JSON/CSV/plot reporting helpers
├── scripts/
│   ├── attack_breakdown.py
│   ├── cache_ssl_embeddings.py
│   ├── check_ssl_ready.py
│   ├── eda.py
│   ├── run_project.py
│   ├── run_robustness_eval.py
│   ├── run_score_fusion.py
│   ├── summarize_results.py
│   ├── train_eval.py
│   ├── train_neural.py
│   └── train_ssl_head.py
├── tests/
│   ├── test_evaluate.py
│   ├── test_feature_cache.py
│   ├── test_features.py
│   ├── test_neural.py
│   ├── test_robustness_eval.py
│   ├── test_score_fusion.py
│   └── test_ssl_baseline.py
├── references/
│   └── standard_baselines.json
├── results/
│   ├── baseline/
│   ├── eda/
│   ├── fusion/
│   ├── neural/
│   └── robustness/
└── archive/
    ├── legacy_experiments_2026-05-06/
    └── original_rf_pipeline_2026-05-05/
```
