# Run Artifact Contract

Every new experiment should write a self-contained run folder. This keeps
training, evaluation, plotting, RAG indexing, and app display loosely coupled.

## Folder Layout

```text
results/<family>/<run_id>/
├── config.json
├── environment.json
├── metrics.json
├── per_attack.csv
├── model_card.md
├── plots/
│   ├── roc.png
│   ├── score_distribution.png
│   └── per_attack_eer.png
├── scores.csv       # ignored if large
└── checkpoints/     # ignored
```

## Required Fields

`metrics.json` should include:

- `run_id`
- `track`
- `model_name`
- `model_family`
- `feature_or_input`
- `external_pretraining`
- `config_path`
- `commit`
- `seed`
- `splits`
- per-split metrics:
  - `eer`
  - `threshold`
  - `accuracy`
  - `n_samples`
  - `n_bonafide`
  - `n_spoof`

## Results Ledger Update

After a run is accepted for reporting:

1. Add the result to `results.md`.
2. Link the run folder.
3. State whether it is protocol-comparable or external-pretrained/applied.
4. Add the exact command used.
5. Add any caveats.

## RAG Readiness

The report assistant should be able to answer from:

- `results.md`
- `metrics.json`
- `per_attack.csv`
- `model_card.md`
- selected plots

If a result is not in those artifacts, the assistant should not claim it.
