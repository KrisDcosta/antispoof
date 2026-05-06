# Incomplete exploratory artifacts

This directory contains generated artifacts from interrupted exploratory runs.
They are preserved for traceability but are not part of the reported baseline
tables.

- `gmm_wcqcc_64c_diag_std_300000frames_seed42.joblib`: trained WCQCC GMM from
  an interrupted full dev/eval scoring run. WCQCC scoring was much slower than
  MFCC/CQCC and should be rerun explicitly only after feature caching or batch
  scoring is added.
