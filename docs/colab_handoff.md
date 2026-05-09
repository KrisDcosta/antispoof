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

### Option A: Restore Dataset From Google Drive

If the dataset archive is already in Google Drive, mount Drive and extract it
into Colab local disk:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
mkdir -p /content/asvspoof_data /content/antispoof/data
cp "/content/drive/MyDrive/asvspoof/LA_clean.tar.zst" /content/asvspoof_data/
zstd -t /content/asvspoof_data/LA_clean.tar.zst
tar --use-compress-program=unzstd -xf /content/asvspoof_data/LA_clean.tar.zst \
  -C /content/antispoof/data
```

If using an already extracted Google Drive dataset instead, symlink it into the
repo:

```bash
mkdir -p data
ln -s /content/drive/MyDrive/<path-to-ASVspoof>/LA data/LA
```

### Option B: Recreate And Transfer The Clean Local Archive

Use this when Drive does not have enough space for the full dataset. From the
local repo on the Mac, create and verify a clean archive from the already
extracted dataset:

```bash
cd /Users/krisdcosta/UCSD/Projects/SAP
mkdir -p tmp
tar -cf - -C data LA | zstd -T0 -19 -o tmp/LA_clean.tar.zst
zstd -t tmp/LA_clean.tar.zst
```

Serve the archive from the Mac. A range-capable server is preferred for large
downloads:

```bash
python -m pip install RangeHTTPServer
cd /Users/krisdcosta/UCSD/Projects/SAP/tmp
python -m RangeHTTPServer 8000
```

In another local terminal, expose the server:

```bash
cloudflared tunnel --url http://localhost:8000
```

In Colab, download from the printed Cloudflare URL, verify, and extract:

```bash
mkdir -p /content/asvspoof_data /content/antispoof/data
wget -O /content/asvspoof_data/LA_clean.tar.zst \
  "https://<cloudflare-url>/LA_clean.tar.zst"
zstd -t /content/asvspoof_data/LA_clean.tar.zst
tar --use-compress-program=unzstd -xf /content/asvspoof_data/LA_clean.tar.zst \
  -C /content/antispoof/data
```

After extraction, the expected path is:

```text
/content/antispoof/data/LA/
```

If the repo clone is elsewhere, copy or symlink that folder to the clone's
`data/LA` path before running training.

## Smoke Run

Run this before any full training:

```bash
python scripts/train_neural.py --config configs/neural_lcnn_smoke.json
```

For the Phase 2 AASIST-lite waveform path:

```bash
python scripts/train_neural.py --config configs/neural_aasist_lite_smoke.json
```

For the Phase 3 external-pretrained/applied SSL path:

```bash
python scripts/check_phase3_ssl_ready.py --config configs/neural_ssl_wavlm_frozen_smoke.json
python scripts/cache_ssl_embeddings.py --config configs/neural_ssl_wavlm_frozen_smoke.json
python scripts/check_phase3_ssl_ready.py --config configs/neural_ssl_wavlm_frozen_smoke.json --check-caches
python scripts/train_ssl_head.py --config configs/neural_ssl_wavlm_frozen_smoke.json
```

Expected output root:

```text
results/neural/<run_id>/
```

## Full Run

```bash
python scripts/train_neural.py --config configs/neural_lcnn.json
```

For the Phase 2 AASIST-lite waveform run:

```bash
python scripts/train_neural.py --config configs/neural_aasist_lite.json
```

For the Phase 3 external-pretrained/applied SSL run:

```bash
python scripts/check_phase3_ssl_ready.py --config configs/neural_ssl_wavlm_frozen.json
python scripts/cache_ssl_embeddings.py --config configs/neural_ssl_wavlm_frozen.json
python scripts/check_phase3_ssl_ready.py --config configs/neural_ssl_wavlm_frozen.json --check-caches
python scripts/train_ssl_head.py --config configs/neural_ssl_wavlm_frozen.json
```

Phase 3 caches pooled mean+std vectors, not full SSL frame sequences. Results
from this track must be labeled external-pretrained/applied and not
protocol-comparable. See `docs/phase3_gpu_runbook.md` for the bundled smoke,
full-run, and acceptance sequence.

## Archiving Full Run Artifacts

Do not commit checkpoints or raw score files. Archive them before disconnecting
Colab:

```bash
tar -czf /content/<run_id>_artifacts.tar.gz \
  -C /content/antispoof/results/neural \
  <run_id>
```

Download directly:

```python
from google.colab import files
files.download("/content/<run_id>_artifacts.tar.gz")
```

If direct browser download fails, copy the small artifact archive to Drive:

```python
from google.colab import drive
drive.mount("/content/drive", force_remount=True)
```

```bash
mkdir -p "/content/drive/MyDrive/asvspoof_artifacts"
cp -v /content/<run_id>_artifacts.tar.gz \
  "/content/drive/MyDrive/asvspoof_artifacts/"
ls -lh "/content/drive/MyDrive/asvspoof_artifacts/"
```

Flush before disconnecting if the Drive web UI lags:

```python
from google.colab import drive
drive.flush_and_unmount()
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
