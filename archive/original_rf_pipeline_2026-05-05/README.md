# Speech Anti-Spoofing — ASVspoof 2019 LA

Detects synthesized/replayed speech vs. genuine using ASVspoof 2019 Logical Access dataset.

Novel feature: **WCQCC** (word-level weighted constant-Q cepstral coefficients) + **AZCR** (average zero-crossing rate on silence segments), compared against MFCC and CQCC baselines.

EER (Equal Error Rate) is the primary metric — computed from classifier probability scores, not binary predictions.

---

## Results (dev set, RandomForest, ASVspoof 2019 LA)

Train: 25,380 files (2,580 bonafide / 22,800 spoof, A01–A06)
Dev: 24,844 files (2,548 bonafide / 22,296 spoof, A01–A06)

| Feature | Accuracy | EER |
|---|---|---|
| MFCC (baseline) | 91.01% | 21.80% |
| CQCC | 89.89% | 23.53% |
| WCQCC | 90.05% | 22.22% |
| WCQCC + AZCR | 90.08% | **21.89%** |

WCQCC+AZCR matches MFCC accuracy while using a more principled feature design.
Full per-attack breakdown and t-SNE plots in `results/`.

---

## Dataset

Download ASVspoof 2019 LA from the [official challenge page](https://www.asvspoof.org/index2019.html).

Expected layout:

```
data/LA/
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
├── ASVspoof2019_LA_eval/flac/
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

`data/` is gitignored — place it at the project root.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

Single feature:
```bash
python scripts/train_eval.py --data data/LA --feature wcqcc
```

Full ablation (all 4 features, comparison plot):
```bash
python scripts/train_eval.py --data data/LA --ablation --output results
```

Quick smoke test (100 samples per split):
```bash
python scripts/train_eval.py --data data/LA --ablation --limit 100
```

---

## Project structure

```
SAP/
├── requirements.txt
├── src/
│   ├── dataset.py      # protocol-file loader — LA only, no random splits
│   ├── features.py     # MFCC, CQCC, WCQCC, AZCR extractors
│   └── evaluate.py     # EER from predict_proba(), ROC, confusion matrix
├── scripts/
│   └── train_eval.py   # train + evaluate, single feature or full ablation
└── results/            # ROC curves, ablation plot saved here
```

---

## What makes WCQCC different

Standard CQCC applies DCT to log-CQT uniformly. WCQCC applies linearly decreasing weights (1.0 → 0.7) across frequency bins before the DCT. This down-weights upper-frequency content where natural speech energy is low but TTS/vocoder artifacts are concentrated, making the resulting cepstrum more discriminative for spoof detection.
