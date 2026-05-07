# Speech Anti-Spoofing on ASVspoof 2019 Logical Access

This project builds a reproducible speech anti-spoofing system for the
ASVspoof 2019 Logical Access task. It follows the classical countermeasure
recipe used in strong challenge systems: extract frame-level cepstral features,
train separate Gaussian mixture models for bonafide and spoofed speech, and
score each utterance with a log-likelihood ratio.

The result is an end-to-end audio ML workflow with dataset validation, EDA,
feature extraction, model training, scoring, EER evaluation, per-attack
diagnostics, and comparison against published ASVspoof reference systems.

Project control documents:

- `plan.md`: roadmap, architecture, phase gates, and implementation rules
- `results.md`: authoritative results ledger for reported numbers

## Method

The classifier is a dual-GMM log-likelihood-ratio system:

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

## Results

Full dev/eval runs were completed on the official ASVspoof 2019 LA protocol.
Eval is the main generalization split because attacks A07-A19 are unseen during
training.

| Method | Dev EER | Eval EER | Interpretation |
|---|---:|---:|---|
| LFCC GMM-LLR | 0.06% | 10.25% | Strongest project result; close to the published LFCC-GMM eval reference |
| CQCC GMM-LLR | 11.15% | 11.59% | Protocol-correct, but uses a simplified Python CQCC extractor |
| MFCC GMM-LLR | 9.77% | 16.41% | Fits dev better than CQCC, generalizes worse on unseen eval attacks |

Published ASVspoof 2019 LA reference systems:

| Reference System | Dev EER | Eval EER | Source |
|---|---:|---:|---|
| Official LFCC-GMM | 2.71% | 8.09% | [ASVspoof 2019 challenge paper](https://arxiv.org/abs/1904.05441) |
| Official CQCC-GMM | 0.43% | 9.57% | [ASVspoof 2019 challenge paper](https://arxiv.org/abs/1904.05441) |

The LFCC system is 2.16 percentage points above the published LFCC-GMM eval
reference. The gap is small enough to make the implementation useful as a
serious classical countermeasure, while also showing the importance of exact
feature recipes and unseen-attack evaluation.

Generated reports:

- `results.md`
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
│   └── asvspoof2019_smoke.json
├── plan.md
├── results.md
├── src/
│   ├── dataset.py        # official ASVspoof protocol loader
│   ├── evaluate.py       # EER, ROC, confusion matrix utilities
│   ├── feature_cache.py  # versioned frame feature cache
│   ├── features.py       # LFCC/MFCC/CQCC/WCQCC extractors
│   ├── gmm_baseline.py   # dual-GMM training and LLR scoring
│   └── reporting.py      # JSON/CSV/plot reporting helpers
├── scripts/
│   ├── run_project.py
│   ├── eda.py
│   ├── train_eval.py
│   ├── attack_breakdown.py
│   └── summarize_results.py
├── tests/
│   ├── test_evaluate.py
│   ├── test_feature_cache.py
│   └── test_features.py
├── references/
│   └── standard_baselines.json
├── results/
│   ├── baseline/
│   └── eda/
└── archive/
    ├── legacy_experiments_2026-05-06/
    └── original_rf_pipeline_2026-05-05/
```
