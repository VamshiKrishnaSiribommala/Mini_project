# Classification Evaluation Protocol: LungXpert

## Averaging Conventions
- **Macro AUROC:** Unweighted arithmetic mean of per-class Area Under the ROC Curve across all 43 binary targets.
- **Macro PR-AUC:** Unweighted arithmetic mean of Average Precision across all 43 classes.
- **Macro F1:** Arithmetic mean of per-class F1-scores evaluated at fixed operating threshold tau=0.50.
- **Micro F1:** Globally pooled precision and recall across all class decisions.
- **Subset Accuracy:** Exact-match ratio where all 43 binary predictions match ground truth simultaneously.
