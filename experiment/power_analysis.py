import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import NormalIndPower

PLOTS_DIR = Path(__file__).parent.parent / "plots"
ALPHA = 0.05
POWER = 0.80


def run_power_analysis(p_control: float, p_variant: float) -> dict:
    effect = proportion_effectsize(p_variant, p_control)
    analysis = NormalIndPower()
    n_required = analysis.solve_power(effect_size=effect, alpha=ALPHA, power=POWER, alternative="two-sided")

    sample_sizes = np.arange(500, int(n_required * 2.5), 100)
    powers = [
        analysis.solve_power(effect_size=effect, nobs1=n, alpha=ALPHA, alternative="two-sided")
        for n in sample_sizes
    ]

    _save_power_curve(sample_sizes, powers, n_required)

    return {
        "p_control": p_control,
        "p_variant": p_variant,
        "effect_size": round(effect, 4),
        "n_required_per_variant": int(np.ceil(n_required)),
        "alpha": ALPHA,
        "power": POWER,
    }


def _save_power_curve(sample_sizes, powers, n_required):
    PLOTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sample_sizes, powers, color="#2563eb", lw=2)
    ax.axhline(POWER, color="#dc2626", ls="--", lw=1.2, label=f"Target power = {POWER}")
    ax.axvline(n_required, color="#16a34a", ls="--", lw=1.2, label=f"Required n ≈ {int(np.ceil(n_required)):,}")
    ax.set_xlabel("Sample size per variant")
    ax.set_ylabel("Statistical power")
    ax.set_title("Power Analysis — Sample Size vs Power")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "power_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from experiment.loader import load_experiment_data
    df = load_experiment_data()
    ctrl = df[df["group"] == "control"]
    var = df[df["group"] == "variant"]
    result = run_power_analysis(ctrl["converted"].mean(), var["converted"].mean())
    for k, v in result.items():
        print(f"  {k}: {v}")
