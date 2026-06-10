# Checkout Flow A/B Test — Bayesian Inference & Hypothesis Testing

A dual-framework experiment analysis pipeline applied to e-commerce checkout flow data. Runs a full A/B test end-to-end: power analysis → frequentist z-test → Bayesian Beta-Binomial model → guardrail monitoring → launch decision.

**Dataset:** Online Shoppers Purchasing Intention (UCI / Kaggle, 12,330 sessions)  
**Primary metric:** `Revenue == True` — visitor completed a purchase  
**Inspired by:** *Experimenting Fast and Slow*, Meta — KDD 2025

---

## Results Summary

| Metric | Value |
|---|---|
| Control conversion rate | 15.05% |
| Variant conversion rate | 15.90% |
| Relative lift | +5.63% |
| p-value (z-test) | 0.193 — not significant |
| P(variant > control) | 90.2% |
| Expected loss | 0.000302 (threshold: 0.0005) |
| **Recommendation** | **MONITOR** |

The frequentist test does not reach significance at α=0.05 — the dataset is underpowered for this effect size (required n ≈ 28,597 per variant vs. ~6,200 available). The Bayesian model, however, finds a 90.2% probability the variant outperforms control and an expected loss well below the stopping threshold, indicating a strong directional signal worth monitoring post-launch.

---

## Methodology

### Frequentist
- Two-proportion z-test at α=0.05 (two-sided), per pre-registered analysis plan
- 95% confidence interval on the absolute difference
- Fixed-horizon design with pre-registered sample size

### Bayesian
- Beta-Binomial conjugate model with `Beta(1,1)` uninformative prior
- Posterior samples via Monte Carlo (100k draws)
- **Expected loss** as a principled stopping rule: `E[max(0, θ_control − θ_variant)] < 0.0005`
- 95% Highest Density Interval (HDI) for each variant

The expected loss stopping rule avoids the Type I error inflation that sequential frequentist peeking introduces, making early stopping statistically sound.

### Guardrail Metrics
Three secondary metrics protect against unintended side effects:

| Guardrail | Direction | Result |
|---|---|---|
| Session duration | Drop is bad | FLAG (−6.49%, p=0.017) |
| Bounce rate | Rise is bad | PASS (−2.57%, p=0.509) |
| Page value | Drop is bad | PASS (−1.63%, p=0.772) |

---

## Project Structure

```
.
├── data/
│   └── online_shoppers_intention.csv   # UCI dataset — 12,330 e-commerce sessions
├── experiment/
│   ├── loader.py           # Data loading and A/B group assignment
│   ├── power_analysis.py   # Sample size and power curve
│   ├── frequentist.py      # Two-proportion z-test
│   ├── bayesian.py         # Beta-Binomial posterior, expected loss
│   ├── guardrails.py       # Secondary metric monitoring
│   └── report.py           # Terminal experiment brief
├── notebooks/
│   └── checkout_ab_test.ipynb   # Full walkthrough with visualizations
├── plots/                  # Generated charts (power curve, posteriors, guardrails)
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

## Running the Analysis

**Notebook (recommended):**
```bash
jupyter notebook notebooks/checkout_ab_test.ipynb
```

**Each module can also be run standalone:**
```bash
python -m experiment.power_analysis
```

---

## Visualizations

The pipeline generates four plots saved to `plots/`:

- **`power_curve.png`** — Statistical power vs. sample size, with required n marked
- **`posterior.png`** — Overlaid Beta posteriors for control and variant
- **`expected_loss.png`** — Loss curve over accumulating data (stopping rule)
- **`guardrails_dashboard.png`** — 2×2 panel comparing all guardrail metrics

---

## Key Takeaways

| Dimension | Frequentist | Bayesian |
|---|---|---|
| Decision signal | Binary (p < α) | Continuous P(variant > control) |
| Uncertainty | Confidence interval | Credible interval (probabilistic) |
| Stopping | Fixed horizon only | Expected loss threshold — flexible |
| Early stopping | Inflates Type I error | No inflation — loss-based |

Running both frameworks side-by-side surfaces cases where a binary p-value would block a launch that a richer probabilistic signal supports — exactly the trade-off highlighted in Meta's KDD 2025 work.
