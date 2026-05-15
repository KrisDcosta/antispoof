# Reproducibility Guide

This guide records how to reproduce or audit the committed project snapshot.
It separates local checks from full GPU runs because the ASVspoof audio,
checkpoints, raw score CSVs, and WavLM caches are too large for git.

## Repository Contents

Committed:

- source code under `src/`
- experiment scripts under `scripts/`
- configs under `configs/`
- unit tests under `tests/`
- documentation under `docs/`
- compact metrics and summary artifacts under `results/**/metrics/` and
  `results/**/summary/`

Not committed:

- `data/LA/`
- full neural run folders
- checkpoints
- raw score CSVs
- SSL embedding caches
- corrupted audio generated during robustness evaluation
- compressed Colab artifact archives

These exclusions are controlled by `.gitignore`.

## Environment Setup

Use Python 3.11 or newer. The local development environment used Python 3.13,
while Colab runs used Python 3.12.

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-neural.txt
```

Run the unit test suite:

```bash
make test
```

Expected current test status:

```text
Ran 34 tests
OK
```

## Dataset Layout

Download ASVspoof 2019 LA from the official challenge source and place it at:

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

Expected protocol counts:

| Split | Total | Bonafide | Spoof |
|---|---:|---:|---:|
| Train | 25,380 | 2,580 | 22,800 |
| Dev | 24,844 | 2,548 | 22,296 |
| Eval | 71,237 | 7,355 | 63,882 |

## Local Validation

These commands are intended to run locally without a full GPU training job:

```bash
make test
make smoke
make neural-smoke
make aasist-smoke
make ssl-ready
```

The smoke runs validate code paths and artifact writing. Their EER values are
not benchmark results because they use small capped samples and short training.

## Classical Baselines

Run the full classical GMM pipeline:

```bash
make project
```

Equivalent direct command:

```bash
python scripts/run_project.py --config configs/asvspoof2019_gmm.json
```

Primary outputs:

```text
results/baseline/metrics/
results/baseline/plots/
results/baseline/summary/
```

## Neural GPU Runs

Full neural runs are GPU-oriented and were executed in Colab.

LCNN:

```bash
python scripts/train_neural.py --config configs/neural_lcnn.json
```

AASIST-lite:

```bash
python scripts/train_neural.py --config configs/neural_aasist_lite.json
```

Frozen WavLM:

```bash
python scripts/cache_ssl_embeddings.py --config configs/neural_ssl_wavlm_frozen.json
python scripts/train_ssl_head.py --config configs/neural_ssl_wavlm_frozen.json
```

The WavLM run requires Hugging Face model download access for
`microsoft/wavlm-base-plus`. Using a Hugging Face token is recommended for
rate-limit stability but is not required for the committed result.

## Score Fusion

Phase 4 fusion requires the LCNN and WavLM raw score CSVs:

```text
results/neural/lcnn_logmel_full_seed42_30ep/scores/dev_scores.csv
results/neural/lcnn_logmel_full_seed42_30ep/scores/eval_scores.csv
results/neural/ssl_wavlm_pooled_full_seed42_50ep/scores/dev_scores.csv
results/neural/ssl_wavlm_pooled_full_seed42_50ep/scores/eval_scores.csv
```

After restoring those artifacts:

```bash
make phase4-fusion
```

The runner aligns by `file_id`, verifies labels and attack IDs, fits
normalization on dev only, selects the fusion rule on dev only, then applies the
frozen rule to eval.

Accepted rule:

```text
fused_score = 0.7 * z_lcnn + 0.3 * z_wavlm
```

## Robustness Evaluation

Phase 5 requires eval audio plus the accepted LCNN, WavLM, and fusion artifacts.
It does not require train/dev audio unless you are rebuilding earlier phases.

Run a smoke check first:

```bash
make robustness-smoke
```

Run the full eval corruption sweep:

```bash
make robustness-eval
```

Primary compact output:

```text
results/robustness/metrics/phase5_eval_corruptions_summary.json
```

The accepted full run took 13,824.6 seconds in Colab and completed clean,
gain, clipping, resampling, and additive-noise conditions. The optional codec
condition was recorded as skipped.

## Accepted Result Evidence

The project claims in `README.md` and `results.md` should trace to these files:

| Claim | Evidence |
|---|---|
| LCNN eval EER 5.67% | `results/neural/metrics/lcnn_logmel_full_seed42_30ep_summary.json` |
| AASIST-lite eval EER 10.64% | `results/neural/metrics/aasist_lite_waveform_seed42_100ep_summary.json` |
| WavLM eval EER 5.08% | `results/neural/metrics/ssl_wavlm_pooled_full_seed42_50ep_summary.json` |
| Fusion eval EER 3.62% | `results/fusion/metrics/lcnn_wavlm_score_fusion_seed42_summary.json` |
| Robustness sweep | `results/robustness/metrics/phase5_eval_corruptions_summary.json` |

## Reporting Guardrails

- Use eval EER as the headline metric.
- Report dev EER as diagnostic only.
- Keep protocol-comparable results separate from external-pretrained/applied
  results.
- Do not tune or select models on eval.
- Do not refit score normalization on eval corruptions.
- Do not call the project SOTA without a matched comparison protocol.
