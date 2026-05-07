# Model Card: <model-name>

## Summary

- Run ID:
- Track: protocol-comparable or applied/external-pretrained
- Model family:
- Input representation:
- Dataset:
- Training split:
- Evaluation splits:
- External pretrained components:

## Intended Use

Describe the intended research, diagnostic, or demo use. Do not claim legal,
forensic, or production certainty unless separately validated.

## Training Configuration

- Config path:
- Commit:
- Seed:
- Device:
- Epochs:
- Batch size:
- Optimizer:
- Learning rate:
- Weight decay:
- Checkpoint path:

## Results

| Split | EER | Accuracy | Notes |
|---|---:|---:|---|
| Dev | pending | pending | |
| Eval | pending | pending | |

Per-attack metrics:

- Path:

Plots:

- ROC:
- score distribution:
- per-attack EER:

## Robustness

List robustness results if evaluated. Keep clean ASVspoof eval separate from
corrupted eval.

## Explainability

List explanation methods and representative artifacts.

## Limitations

- Eval attacks are the primary generalization test.
- Accuracy is secondary because the dataset is class-imbalanced.
- External pretrained models must be labeled as not strictly challenge-comparable.
- The detector should not be treated as definitive forensic proof.

## Reproduction

```bash
python scripts/train_neural.py --config <config-path>
```
