# Colab / VSCode GPU Handoff

Use this when running neural training on Colab Pro GPU while keeping the repo
source of truth in GitHub.

## Assumptions

- Repository: `KrisDcosta/antispoof`
- Dataset is available in Colab storage or mounted Google Drive.
- Large outputs, checkpoints, caches, and score CSVs are not committed.
- Lightweight metrics, summaries, selected plots, and model cards can be copied
  back into the repo and committed.

## Setup

```bash
git clone git@github.com:KrisDcosta/antispoof.git
cd antispoof
python -m pip install --upgrade pip
python -m pip install -r requirements-neural.txt
```

If SSH is not configured in Colab, use HTTPS:

```bash
git clone https://github.com/KrisDcosta/antispoof.git
```

## Dataset Placement

Expected layout:

```text
data/LA/
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
├── ASVspoof2019_LA_eval/flac/
└── ASVspoof2019_LA_cm_protocols/
```

If using Google Drive, symlink the dataset into the repo:

```bash
mkdir -p data
ln -s /content/drive/MyDrive/<path-to-ASVspoof>/LA data/LA
```

## Smoke Run

Run this before any full training:

```bash
python scripts/train_neural.py --config configs/neural_lcnn_smoke.json
```

Expected output root:

```text
results/neural/<run_id>/
```

## Full Run

```bash
python scripts/train_neural.py --config configs/neural_lcnn.json
```

## Returning Results To Repo

Copy back only:

- `metrics.json`
- `per_attack.csv`
- selected plots
- `model_card.md`
- config used
- updated `results.md`

Do not commit:

- dataset audio
- checkpoints
- embedding caches
- raw score CSVs if large
- API keys

## Validation Before Commit

```bash
python -m unittest discover -s tests
python -m compileall src scripts tests
git status --short --ignored
```

Confirm that ignored files include checkpoints, scores, caches, and dataset
paths before pushing.
