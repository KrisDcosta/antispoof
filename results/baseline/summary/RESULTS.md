# ASVspoof 2019 LA Results

## Project Results

| Method | Dev EER | Eval EER |
|---|---:|---:|
| CQCC GMM-LLR | 11.15% | 11.59% |
| LFCC GMM-LLR | 0.06% | 10.25% |
| MFCC GMM-LLR | 9.77% | 16.41% |

## Published Reference Systems

| Reference System | Dev EER | Eval EER | Source |
|---|---:|---:|---|
| Official LFCC-GMM | 2.71% | 8.09% | [ASVspoof 2019 evaluation plan and challenge paper](https://arxiv.org/abs/1904.05441) |
| Official CQCC-GMM | 0.43% | 9.57% | [ASVspoof 2019 evaluation plan and challenge paper](https://arxiv.org/abs/1904.05441) |

## Notes

- Best project eval result: LFCC GMM-LLR at 10.25% EER.
- LFCC GMM-LLR is 2.16 percentage points above the matching published eval reference.
- LFCC is the current strongest project feature; eval remains harder than dev because eval attacks are unseen.
- Project CQCC uses the repository's simplified librosa CQT + DCT extraction.
- Published references use the official ASVspoof 2019 recipes.
- Eval is the primary generalization split because attacks A07-A19 are unseen during training.
