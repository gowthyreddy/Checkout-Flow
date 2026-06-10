import numpy as np
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

ALPHA = 0.05


def run_frequentist(control_conversions: int, control_n: int,
                    variant_conversions: int, variant_n: int) -> dict:
    counts = np.array([variant_conversions, control_conversions])
    nobs = np.array([variant_n, control_n])

    z_stat, p_value = proportions_ztest(counts, nobs, alternative="two-sided")

    p_ctrl = control_conversions / control_n
    p_var = variant_conversions / variant_n
    diff = p_var - p_ctrl

    # 95% CI on the difference via normal approximation
    se = np.sqrt(p_ctrl * (1 - p_ctrl) / control_n + p_var * (1 - p_var) / variant_n)
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se

    return {
        "p_control": round(p_ctrl, 5),
        "p_variant": round(p_var, 5),
        "absolute_lift": round(diff, 5),
        "relative_lift_pct": round(diff / p_ctrl * 100, 2),
        "z_stat": round(z_stat, 4),
        "p_value": round(p_value, 6),
        "ci_95": (round(ci_low, 5), round(ci_high, 5)),
        "significant": p_value < ALPHA,
    }
