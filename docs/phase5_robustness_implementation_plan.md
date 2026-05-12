# Phase 5 Robustness Evaluation Implementation Plan

This file is a handoff plan for a Codex implementation chat working in:

```text
/Users/krisdcosta/UCSD/Projects/SAP
```

The goal is to implement Phase 5 without changing the accepted Phase 1-4
results.

## Critical Context

Current accepted results:

| System | Track | Eval EER |
|---|---|---:|
| LCNN log-mel | protocol-comparable | 5.67% |
| frozen WavLM pooled MLP | external-pretrained/applied | 5.08% |
| LCNN+WavLM score fusion | external-pretrained/applied | 3.62% |

Phase 4 selected fusion rule:

```text
fused_score = 0.7 * z_lcnn + 0.3 * z_wavlm
```

Use the Phase 4 dev normalization statistics from:

```text
results/fusion/metrics/lcnn_wavlm_score_fusion_seed42_summary.json
```

Important repo state:

- `main` is pushed through Phase 4.
- `scripts/dud.ipynb` is an unrelated untracked file; do not touch it.
- Dataset audio, restored checkpoints, raw score CSVs, corrupted audio, and
  full run folders should stay out of git.
- Commit compact summaries and docs only after a full accepted result.

## Phase 5 Research Question

Do LCNN, frozen WavLM, and LCNN+WavLM fusion remain reliable under realistic
consumer-audio distortions?

This phase should evaluate robustness, not train new models.

## Scope

Evaluate these systems:

1. `lcnn_logmel_full_seed42_30ep`
2. `ssl_wavlm_pooled_full_seed42_50ep`
3. `lcnn_wavlm_score_fusion_seed42`

Start with eval split only.

Corruption families:

```text
clean
gain
clipping
resampling
additive_noise
codec
```

Recommended severities:

```text
gain: -12dB, -6dB, +6dB
clipping: 0.8, 0.6, 0.4
resampling: 12000Hz, 8000Hz
noise: 20dB, 10dB, 5dB SNR
codec: opus/mp3 high, medium, low bitrate if local tooling supports it
```

If codec tooling is unavailable or brittle, implement all other corruptions
first and make codec optional with a clear skip reason.

## Why These Decisions

Use eval only:

- Phase 5 measures generalization under channel shift.
- No retraining or hyperparameter tuning should happen here.

Use frozen Phase 4 fusion rule:

- Fusion already selected weights using dev.
- Refitting fusion on corrupted eval would leak test distribution information.

Use synthetic corruptions:

- They provide controlled stress tests for consumer-audio conditions.
- They are not a full real-world robustness claim, so docs must state that
  limitation.

Use per-condition and per-attack EER:

- Overall EER answers "which model is robust?"
- Per-attack EER answers "which spoof families fail under distortion?"

## Expected New Files

Prefer one orchestrator first:

```text
scripts/run_robustness_eval.py
configs/robustness_phase5_smoke.json
configs/robustness_phase5.json
tests/test_robustness_phase5.py
docs/phase5_robustness_plan.md
```

Add Makefile helpers:

```make
robustness-smoke:
	$(PYTHON) scripts/run_robustness_eval.py --config configs/robustness_phase5_smoke.json

robustness-eval:
	$(PYTHON) scripts/run_robustness_eval.py --config configs/robustness_phase5.json
```

## Artifact Layout

Full local artifact folder:

```text
results/robustness/phase5_eval_corruptions/
  config.json
  metrics.json
  corruption_manifest.csv
  per_condition_metrics.csv
  per_attack_condition_metrics.csv
  model_card.md
  plots/
    eer_by_condition.png
    relative_degradation.png
    per_attack_heatmap.png
  scores/
    lcnn_scores.csv
    wavlm_scores.csv
    fusion_scores.csv
```

Committed compact summary:

```text
results/robustness/metrics/phase5_eval_corruptions_summary.json
```

Update `.gitignore` so full robustness run folders and corrupted audio are not
committed, but `results/robustness/metrics/**` remains committable.

## Required Inputs

Dataset:

```text
data/LA/ASVspoof2019_LA_eval/flac/
data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
```

LCNN artifacts:

```text
results/neural/lcnn_logmel_full_seed42_30ep/checkpoints/best.pt
results/neural/lcnn_logmel_full_seed42_30ep/config.json
```

WavLM artifacts:

```text
results/neural/ssl_wavlm_pooled_full_seed42_50ep/checkpoints/best.pt
results/neural/ssl_wavlm_pooled_full_seed42_50ep/config.json
```

Phase 4 fusion summary:

```text
results/fusion/metrics/lcnn_wavlm_score_fusion_seed42_summary.json
```

If model checkpoints are not local, stop and tell the user exactly which
artifact archives need to be restored. Do not attempt to retrain.

## Implementation Steps

### 1. Inspect Existing Scoring APIs

Read existing code before implementing:

```text
scripts/train_neural.py
scripts/cache_ssl_embeddings.py
scripts/train_ssl_head.py
scripts/run_score_fusion.py
src/neural/
src/evaluate.py
src/reporting.py
```

Reuse existing:

- protocol loading
- waveform crop/pad policy
- model loading
- score CSV writing
- EER/per-attack metrics
- plot/report helpers where practical

### 2. Add Configs

Smoke config should use:

- tiny deterministic eval cap
- clean + one or two corruptions
- one severity each
- no codec unless easy

Full config should include all selected corruption families and severities.

Example config shape:

```json
{
  "run_id": "phase5_eval_corruptions",
  "data_root": "data/LA",
  "output_root": "results/robustness",
  "split": "eval",
  "sample_rate": 16000,
  "num_samples": 64600,
  "limit": null,
  "systems": {
    "lcnn": {
      "config_path": "results/neural/lcnn_logmel_full_seed42_30ep/config.json",
      "checkpoint_path": "results/neural/lcnn_logmel_full_seed42_30ep/checkpoints/best.pt"
    },
    "wavlm": {
      "config_path": "results/neural/ssl_wavlm_pooled_full_seed42_50ep/config.json",
      "checkpoint_path": "results/neural/ssl_wavlm_pooled_full_seed42_50ep/checkpoints/best.pt"
    },
    "fusion": {
      "summary_path": "results/fusion/metrics/lcnn_wavlm_score_fusion_seed42_summary.json"
    }
  },
  "corruptions": [
    {"name": "clean", "type": "identity"},
    {"name": "gain_m6db", "type": "gain", "db": -6},
    {"name": "clip_0p6", "type": "clipping", "threshold": 0.6},
    {"name": "resample_8k", "type": "resample", "target_sample_rate": 8000},
    {"name": "noise_10db", "type": "noise", "snr_db": 10}
  ]
}
```

### 3. Implement Audio Corruptions

Implement deterministic tensor-level corruptions where possible:

- gain: multiply waveform by `10 ** (db / 20)`
- clipping: clamp waveform to `[-threshold, threshold]`
- resampling: resample from 16 kHz to target, then back to 16 kHz
- noise: add seeded Gaussian noise at target SNR
- clean: identity

Keep audio in memory for the first implementation. Do not write corrupted audio
unless absolutely needed for debugging, and if written, place it under an ignored
folder.

Codec compression can be optional because it may require `ffmpeg` or external
codec support. Detect support and skip with metadata if unavailable.

### 4. Score Models

For each condition:

1. load original eval waveform
2. apply corruption
3. score with LCNN
4. score with WavLM
5. apply frozen Phase 4 fusion
6. write scores and metrics

Larger score must mean more bonafide for all systems.

Fusion must use Phase 4 dev normalization statistics. Do not normalize using
corrupted eval statistics.

### 5. Metrics

Compute:

```text
EER by model and condition
accuracy at condition EER threshold
relative EER degradation vs clean
per-attack EER by model and condition
fusion gain/loss vs LCNN and WavLM
```

Primary table:

```text
condition, corruption_type, severity, model, eer, delta_vs_clean, ratio_vs_clean
```

Per-attack table:

```text
condition, model, attack_id, eer
```

### 6. Plots

At minimum:

- EER by condition/model bar chart
- relative degradation chart
- per-attack heatmap for selected/important conditions

Keep plot generation robust to missing optional codec rows.

### 7. Tests

Add unit tests for:

- gain corruption changes amplitude as expected
- clipping clamps values
- resampling returns expected sample rate/shape
- noise corruption is deterministic with seed
- fusion uses stored Phase 4 stats, not eval stats
- output artifact writing on synthetic score rows
- config validation catches missing checkpoints

Run:

```bash
./venv/bin/python -m unittest discover -s tests
```

### 8. Smoke Run

Run:

```bash
make robustness-smoke
```

Smoke success means:

- script completes
- score CSVs exist
- metrics JSON exists
- compact summary exists
- plots exist
- no raw audio is committed

Smoke EER is not a result claim.

### 9. Full Run

Run:

```bash
make robustness-eval
```

If full run is expensive, start with the three strongest conditions:

```text
clean
resample_8k
noise_10db
codec_medium or clip_0p6
```

Then expand once validated.

### 10. Acceptance and Docs

Only after a successful full run:

Update:

```text
README.md
results.md
plan.md
docs/phase5_robustness_plan.md
```

Do not claim real-world robustness beyond the tested synthetic corruptions.

## Guardrails

Data leakage:

- Do not train on corrupted eval.
- Do not tune corruption severities after seeing eval results.
- Do not refit normalization on corrupted eval.
- Do not update Phase 4 fusion weights.

Security/storage:

- Do not commit ASVspoof audio.
- Do not commit corrupted audio.
- Do not commit checkpoints.
- Do not commit raw score CSVs if they are large.
- Do not commit private Drive links or tokens.

Validity:

- Keep protocol-comparable LCNN separate from WavLM/fusion applied systems.
- Report negative results.
- Include clean condition in the same run as corrupted conditions.
- Use deterministic seeds for noise.
- Record skipped optional corruptions and why.

## Stop Conditions

Stop and report back if:

- required LCNN/WavLM checkpoints are missing
- score direction is ambiguous
- existing scoring code cannot be reused without major refactor
- codec support requires large dependency changes
- full run would overwrite accepted Phase 1-4 artifacts

## Final Deliverable

At the end of implementation, provide:

1. files changed
2. validation commands and outputs
3. smoke/full result summary
4. whether Phase 5 is accepted or only infrastructure-ready
5. any missing restored artifacts
6. clear next recommendation

