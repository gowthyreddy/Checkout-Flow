import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from pathlib import Path

PLOTS_DIR = Path(__file__).parent.parent / "plots"
N_SAMPLES = 100_000
LOSS_THRESHOLD = 0.0005


def run_bayesian(control_conversions: int, control_n: int,
                 variant_conversions: int, variant_n: int,
                 seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    # Beta(1,1) uninformative prior + data → posterior
    a_ctrl = 1 + control_conversions
    b_ctrl = 1 + (control_n - control_conversions)
    a_var = 1 + variant_conversions
    b_var = 1 + (variant_n - variant_conversions)

    samples_ctrl = rng.beta(a_ctrl, b_ctrl, N_SAMPLES)
    samples_var = rng.beta(a_var, b_var, N_SAMPLES)

    prob_variant_wins = float(np.mean(samples_var > samples_ctrl))
    expected_loss = float(np.mean(np.maximum(0, samples_ctrl - samples_var)))

    hdi_ctrl = np.percentile(samples_ctrl, [2.5, 97.5])
    hdi_var = np.percentile(samples_var, [2.5, 97.5])

    _save_posterior_plot(a_ctrl, b_ctrl, a_var, b_var)
    _save_expected_loss_curve(control_conversions, control_n, variant_conversions, variant_n, seed)

    return {
        "p_control_posterior_mean": round(a_ctrl / (a_ctrl + b_ctrl), 5),
        "p_variant_posterior_mean": round(a_var / (a_var + b_var), 5),
        "prob_variant_wins": round(prob_variant_wins, 4),
        "expected_loss": round(expected_loss, 6),
        "safe_to_stop": expected_loss < LOSS_THRESHOLD,
        "hdi_95_control": (round(hdi_ctrl[0], 5), round(hdi_ctrl[1], 5)),
        "hdi_95_variant": (round(hdi_var[0], 5), round(hdi_var[1], 5)),
    }


def _save_posterior_plot(a_ctrl, b_ctrl, a_var, b_var):
    PLOTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    # Zoom x-axis to the region of interest (±5 std devs around the means)
    mean_ctrl = a_ctrl / (a_ctrl + b_ctrl)
    mean_var  = a_var  / (a_var  + b_var)
    std_ctrl  = (a_ctrl * b_ctrl / ((a_ctrl + b_ctrl) ** 2 * (a_ctrl + b_ctrl + 1))) ** 0.5
    std_var   = (a_var  * b_var  / ((a_var  + b_var)  ** 2 * (a_var  + b_var  + 1))) ** 0.5
    x_min = max(0, min(mean_ctrl, mean_var) - 6 * max(std_ctrl, std_var))
    x_max = min(1, max(mean_ctrl, mean_var) + 6 * max(std_ctrl, std_var))
    x = np.linspace(x_min, x_max, 1000)
    pdf_ctrl = beta_dist.pdf(x, a_ctrl, b_ctrl)
    pdf_var  = beta_dist.pdf(x, a_var,  b_var)

    ax.plot(x, pdf_ctrl, color="#2563eb", lw=2, label="Control posterior")
    ax.plot(x, pdf_var,  color="#dc2626", lw=2, label="Variant posterior")
    ax.fill_between(x, pdf_ctrl, alpha=0.15, color="#2563eb")
    ax.fill_between(x, pdf_var,  alpha=0.15, color="#dc2626")
    ax.set_xlabel("Conversion rate θ")
    ax.set_ylabel("Density")
    ax.set_title("Bayesian Posterior — Beta-Binomial Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "posterior.png", dpi=150)
    plt.close(fig)


def _save_expected_loss_curve(c_conv, c_n, v_conv, v_n, seed):
    rng = np.random.default_rng(seed)
    steps = np.linspace(0.05, 1.0, 40)
    losses = []
    for frac in steps:
        cc = int(c_conv * frac)
        cn = int(c_n * frac)
        vc = int(v_conv * frac)
        vn = int(v_n * frac)
        s_c = rng.beta(1 + cc, 1 + (cn - cc), N_SAMPLES)
        s_v = rng.beta(1 + vc, 1 + (vn - vc), N_SAMPLES)
        losses.append(float(np.mean(np.maximum(0, s_c - s_v))))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps * 100, losses, color="#7c3aed", lw=2)
    ax.axhline(LOSS_THRESHOLD, color="#dc2626", ls="--", lw=1.2,
               label=f"Stop threshold = {LOSS_THRESHOLD}")
    ax.set_xlabel("% of data observed")
    ax.set_ylabel("Expected loss")
    ax.set_title("Expected Loss Over Accumulating Data (Stopping Rule)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "expected_loss.png", dpi=150)
    plt.close(fig)
