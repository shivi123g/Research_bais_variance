# Bias–Variance Trade-off Analysis

A self-initiated empirical research project investigating the theoretical relationship between model complexity, regularization, and generalization error — implemented from scratch and validated on real-world data.

---

## Overview

The bias–variance trade-off is a foundational concept in statistical machine learning: as model complexity increases, bias decreases but variance increases, and the optimal model minimizes their sum (total MSE). This project empirically validates this theory using a real-world dataset, custom model implementations, and bootstrap-based error decomposition.

Two experiments are conducted:

1. **Polynomial Degree vs. Error** — How does increasing model complexity affect bias², variance, and total MSE?
2. **Ridge Regularization (λ) vs. Error** — How does L2 regularization strength trade off bias against variance?

---

## Methodology

### Dataset
- **Sklearn Diabetes Dataset** (real-world, not synthetic) — a standard regression benchmark with 442 samples
- Single feature extracted (`X[:, 2]`) to enable clean polynomial expansion and interpretable visualization
- 70/30 train-test split with fixed random seed for reproducibility

### Models (Implemented from Scratch)
- `LinearRegressionScratch` — closed-form OLS solution via normal equations
- `RidgeRegressionScratch` — L2-regularized regression with tunable λ

Both models are built without scikit-learn's estimators, demonstrating understanding of the underlying mathematics.

### Bias–Variance Decomposition
Error decomposition is performed via **bootstrap sampling** (`n_runs=80`):

```
Bias²  = E[(ȳ_pred - y_true)²]
Variance = E[(y_pred - ȳ_pred)²]
Total MSE ≈ Bias² + Variance + Noise
```

---

## Experiment 1: Model Complexity (Polynomial Degree 1–9)

Polynomial features of increasing degree are constructed manually and fitted using `LinearRegressionScratch`. Bias², Variance, and Total MSE are tracked across degrees.

**Key finding:** On this noisy, low-dimensional real-world dataset, bias remains dominant across most complexity levels. Variance increases meaningfully only beyond degree ~5, reflecting that classical U-shaped curves are more pronounced on synthetic data — a nuance often omitted in textbook treatments.

---

## Experiment 2: Regularization Strength (Ridge, λ ∈ {0, 0.1, 1, 10, 100})

Ridge regression is applied at increasing regularization strengths. Results are plotted on a log scale.

**Key finding:** As λ increases, variance decreases consistently while bias increases — directly validating the theoretical regularization trade-off. This demonstrates that L2 regularization is a controlled mechanism for shifting the bias–variance balance, not merely a heuristic.

---

## Results Summary

### Polynomial Complexity
| Degree | Bias² | Variance | Total MSE |
|--------|-------|----------|-----------|
| 1 | High | Low | Moderate |
| 4–5 | Moderate | Moderate | Optimal zone |
| 8–9 | Low | High | Increases |

### Ridge Regularization
| λ | Bias² | Variance | Total MSE |
|---|-------|----------|-----------|
| 0.0 | Low | High | Higher |
| 1.0 | Moderate | Moderate | Near-optimal |
| 100.0 | High | Low | Higher |



---

## Key Takeaways

- Real-world data does not always produce the textbook U-shaped bias–variance curve — noise and dimensionality matter
- Implementing models from scratch (rather than using sklearn wrappers) reveals the mathematical mechanics behind regularization
- Bootstrap-based decomposition is a rigorous empirical method for isolating bias and variance without closed-form assumptions

