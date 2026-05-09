# Phase 3 SSL Embedding Plan

Phase 3 adds an external-pretrained speech representation track while Phase 2
AASIST-lite GPU training finishes. These runs are useful applied baselines, but
they are not protocol-comparable ASVspoof challenge systems because the speech
encoder is pretrained outside the official training protocol.

## Goal

Evaluate whether frozen self-supervised speech embeddings improve spoof
generalization over the current project baselines:

- LFCC GMM-LLR classical baseline
- log-mel LCNN trained from scratch
- AASIST-lite waveform model trained from scratch

Primary metric remains eval EER, with dev EER used for model selection.

## Initial Scope

Start with a frozen encoder and a shallow supervised head:

- encoder: `microsoft/wavlm-base-plus` first, with `facebook/wav2vec2-base` as
  a fallback if dependency or memory issues appear
- input: mono waveform resampled to 16 kHz
- clip policy: reuse the Phase 2 64,600-sample crop/pad policy for direct run
  comparability
- feature cache: one pooled-vector file per split, keyed by file id and encoder
  name
- cache representation: `concat(mean(last_hidden_state), std(last_hidden_state))`
- classifier: train-normalized pooled-vector MLP
- loss: weighted cross entropy from train label counts
- score: bonafide softmax probability

Do not fine-tune the SSL encoder until the frozen-cache baseline has a complete
dev/eval result and artifact folder. Do not cache full frame sequences for the
first baseline; add a separate frame-cache mode later if attention pooling is
needed.

## Proposed Files

Implementation should stay close to the existing neural experiment interface:

- `configs/neural_ssl_wavlm_frozen_smoke.json`
- `configs/neural_ssl_wavlm_frozen.json`
- `scripts/cache_ssl_embeddings.py`
- `scripts/train_ssl_head.py`
- `src/neural/ssl_embeddings.py`
- `src/neural/ssl_dataset.py`
- `src/neural/ssl_models.py`

The training output should still follow `docs/run_artifact_contract.md`.

## Cache Contract

Write pooled cached embeddings under:

```text
results/cache/ssl/<encoder_slug>/<split>.pt
```

Each cache should contain:

- `encoder_name`
- `encoder_revision`, if available
- `processor_name`
- `transformers_version`
- `sample_rate`
- `num_samples`
- `hidden_state_source`
- `cache_representation`
- `torch_dtype`
- `cache_device`
- `split`
- `items`, with `file_id`, `path`, `label`, `system_id`, and pooled embedding
  tensor
- `created_at`

The cache script should refuse to overwrite an existing cache unless an explicit
flag is provided.

## Run Order

1. Add the SSL dependency path in `requirements-neural.txt`.
2. Validate the Phase 3 config and installed dependencies with
   `scripts/check_phase3_ssl_ready.py`.
3. Implement cache generation for smoke limits first.
4. Run a local or Colab smoke cache with a tiny class-balanced sample limit.
5. Validate cache metadata and pooled vector shapes with
   `scripts/check_phase3_ssl_ready.py --check-caches`.
6. Train the frozen-head smoke config and verify `metrics.json`, per-attack CSV,
   plots, and model card are produced.
7. Run the full cache build on GPU.
8. Validate full cache metadata and pooled vector shapes.
9. Train the full frozen-head baseline on GPU.
10. Update `results.md` with an explicit external-pretraining label.
11. Only then consider top-layer fine-tuning, adapters, or LoRA.

## Acceptance Criteria

The first Phase 3 result is accepted when:

- dev and eval EER are present
- per-attack eval EER is present
- run command and config path are recorded
- cache encoder name and sample policy are recorded
- cache representation is `pooled_mean_std`
- normalization statistics are fit on train only
- class imbalance is handled with weighted cross entropy
- `external_pretraining` is `true` in `metrics.json`
- model card says the result is external-pretrained/applied
- no checkpoint or large cache artifact is committed

## Risks

- Cache size may be large if frame sequences are cached. The first baseline
  caches pooled mean+std vectors only and keeps all caches ignored.
- Encoder version drift can affect results. Pin the Hugging Face model name and
  record revision when available.
- Full fine-tuning may overfit dev or exceed Colab memory. Treat it as a later
  experiment, not the first Phase 3 deliverable.
- The comparison should be worded carefully: Phase 3 is
  external-pretrained/applied and not protocol-comparable. Any comparison
  against ASVspoof challenge baselines must keep that caveat adjacent.
