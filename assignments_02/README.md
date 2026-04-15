# Mini-Project: Predicting Student Math Performance

**Dataset:** UCI Student Performance Dataset — 395 students from two Portuguese secondary schools, 2005–2006.  
**Goal:** predict final math grade (G3, scale 0–20) using student background and behavioral features, using only early-available inputs and avoiding leakage from intermediate grades (G1, G2).

---

## Data Notes

- Separator is `;` not comma — requires `pd.read_csv(..., sep=";")`
- `G3=0` means the student was absent from the final exam, not an actual zero. 38 rows removed.
- `yes`/`no` columns encoded as 1/0; `sex` encoded as F=0, M=1

---

## Results

| Model | Features | Test R² | RMSE |
|-------|----------|---------|------|
| Baseline | `failures` only | 0.09 | 2.96 |
| Full model | 11 features | 0.15 | 2.86 |
| With G1 | 11 features + G1 | 0.75 | — |

RMSE is reported on the 0–20 grading scale.

---

## Key Findings

**`failures` (−0.29 Pearson corr)** — strongest negative predictor. Each additional past failure → ~1.1 points lower grade.

**`schoolsup` (coef −2.06)** — largest negative coefficient, driven by selection bias. Struggling students are more likely to receive support, so the model associates support with lower performance. This is correlation, not causation.

**`Medu` (+0.19 Pearson corr)** — strongest positive signal. Family educational background plays a role.

**`Fedu` vs `Medu`:** Fedu has a stronger regression coefficient despite Medu having a higher simple correlation — likely due to multicollinearity when both are in the model together.

**`absences` correlation:** shifts from +0.03 to −0.21 after filtering G3=0, revealing how data cleaning changes conclusions.

**G1 (target leakage):** adding G1 raises R² from 0.15 → 0.75. This effectively acts as target leakage in real-world usage, since G1 is an earlier measure of the same outcome.

---

## Plots

| File | Description |
|------|-------------|
| `g3_distribution.png` | Histogram of G3 — cluster of zeros visible |
| `g3_by_failures.png` | G3 by number of past failures (box plot) |
| `g3_by_higher.png` | G3 by aspiration for higher education (box plot) |
| `g3_vs_absences.png` | G3 vs absences — weak, noisy negative trend |
| `correlation_matrix.png` | Correlation heatmap across all features |
| `baseline_failures.png` | Baseline model predictions vs actual |
| `predicted_vs_actual.png` | Full model: predicted vs actual on test set |

---

## Final Takeaway

Academic performance is only partially predictable from survey-style features. Even with multiple variables, the model explains a limited portion of variance (R² ≈ 0.15), indicating that many important factors are unobserved.

This project highlights:
- the importance of data cleaning (removing G3=0)
- the risk of misleading correlations (`schoolsup`)
- the impact of feature interactions (`Medu` vs `Fedu`)
- the danger of leakage (G1)

---

## How to Run

```bash
cd assignments_02/
python project_02.py
```
