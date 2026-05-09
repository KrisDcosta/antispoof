# Phase 3 GPU Runbook

Use this when GPU time is available. All non-GPU implementation and synthetic
validation should already be complete before starting this sequence.

## Preflight

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-neural.txt
python -m unittest discover -s tests
python scripts/check_phase3_ssl_ready.py --config configs/neural_ssl_wavlm_frozen.json
```

Expected result:

- tests pass
- package checks pass
- config checks pass
- no WavLM model run has happened yet

## Smoke Run

```bash
python scripts/cache_ssl_embeddings.py --config configs/neural_ssl_wavlm_frozen_smoke.json
python scripts/check_phase3_ssl_ready.py --config configs/neural_ssl_wavlm_frozen_smoke.json --check-caches
python scripts/train_ssl_head.py --config configs/neural_ssl_wavlm_frozen_smoke.json
```

Expected artifacts:

```text
results/cache/ssl/wavlm_base_plus/{train,dev,eval}.pt
results/neural/ssl_wavlm_pooled_smoke_seed42_1ep/
```

Smoke EER is only a pipeline check and should not be reported as a model result.

## Full Run

```bash
python scripts/cache_ssl_embeddings.py --config configs/neural_ssl_wavlm_frozen.json
python scripts/check_phase3_ssl_ready.py --config configs/neural_ssl_wavlm_frozen.json --check-caches
python scripts/train_ssl_head.py --config configs/neural_ssl_wavlm_frozen.json
```

Expected artifacts:

```text
results/cache/ssl/wavlm_base_plus/{train,dev,eval}.pt
results/neural/ssl_wavlm_pooled_full_seed42_50ep/
```

## Acceptance Checklist

Accept the Phase 3 result only if:

- `metrics.json` has dev and eval EER.
- `metrics.json` has `external_pretraining: true`.
- `track` is `external-pretrained/applied`.
- `cache_representation` is `pooled_mean_std`.
- `normalization` is `train_mean_std`.
- `loss` is `weighted_cross_entropy`.
- per-attack eval EER exists.
- `model_card.md` says not protocol-comparable.
- caches, checkpoints, and raw score CSVs remain ignored.

Update `results.md` only after acceptance, using adjacent caveat wording:

```text
external-pretrained/applied, not protocol-comparable
```
