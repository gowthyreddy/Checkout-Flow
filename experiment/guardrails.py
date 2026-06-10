import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

PLOTS_DIR = Path(__file__).parent.parent / "plots"
ALPHA = 0.05


def run_guardrails(ctrl, var) -> dict:
    results = {}

    # Guardrail 1: session_duration (higher = more engagement, drop = bad)
    _, p_dur = stats.ttest_ind(var["session_duration"], ctrl["session_duration"])
    dur_change = (var["session_duration"].mean() - ctrl["session_duration"].mean()) / ctrl["session_duration"].mean()
    results["session_duration"] = {
        "control_mean": round(ctrl["session_duration"].mean(), 3),
        "variant_mean": round(var["session_duration"].mean(), 3),
        "change_pct": round(dur_change * 100, 2),
        "p_value": round(p_dur, 6),
        "status": "FLAG" if (dur_change < -0.03 and p_dur < ALPHA) else "PASS",
    }

    # Guardrail 2: bounce_rate (lower = better, rise = bad)
    _, p_bounce = stats.ttest_ind(var["bounce_rate"], ctrl["bounce_rate"])
    bounce_change = (var["bounce_rate"].mean() - ctrl["bounce_rate"].mean()) / (ctrl["bounce_rate"].mean() + 1e-9)
    results["bounce_rate"] = {
        "control_mean": round(ctrl["bounce_rate"].mean(), 5),
        "variant_mean": round(var["bounce_rate"].mean(), 5),
        "change_pct": round(bounce_change * 100, 2),
        "p_value": round(p_bounce, 6),
        "status": "FLAG" if (bounce_change > 0.03 and p_bounce < ALPHA) else "PASS",
    }

    # Guardrail 3: page_value (higher = better, drop = bad)
    _, p_pv = stats.ttest_ind(var["page_value"], ctrl["page_value"])
    pv_change = (var["page_value"].mean() - ctrl["page_value"].mean()) / (ctrl["page_value"].mean() + 1e-9)
    results["page_value"] = {
        "control_mean": round(ctrl["page_value"].mean(), 3),
        "variant_mean": round(var["page_value"].mean(), 3),
        "change_pct": round(pv_change * 100, 2),
        "p_value": round(p_pv, 6),
        "status": "FLAG" if (pv_change < -0.03 and p_pv < ALPHA) else "PASS",
    }

    _save_dashboard(ctrl, var, results)
    return results


def _save_dashboard(ctrl, var, results):
    PLOTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Guardrail Metrics Dashboard — Checkout A/B Test", fontsize=13, fontweight="bold")

    metrics = [
        ("Conversion Rate", ctrl["converted"].mean(), var["converted"].mean(), "Primary Metric"),
        ("Session Duration (s)", results["session_duration"]["control_mean"],
         results["session_duration"]["variant_mean"],
         f'Guardrail - {results["session_duration"]["status"]}'),
        ("Bounce Rate", results["bounce_rate"]["control_mean"],
         results["bounce_rate"]["variant_mean"],
         f'Guardrail - {results["bounce_rate"]["status"]}'),
        ("Page Value", results["page_value"]["control_mean"],
         results["page_value"]["variant_mean"],
         f'Guardrail - {results["page_value"]["status"]}'),
    ]

    colors = {"PASS": "#16a34a", "FLAG": "#dc2626", "Primary Metric": "#2563eb"}

    for ax, (label, c_val, v_val, subtitle) in zip(axes.flat, metrics):
        color = colors.get(subtitle.split("- ")[-1], "#2563eb")
        bars = ax.bar(["Control", "Variant"], [c_val, v_val],
                      color=["#94a3b8", color], width=0.5)
        ax.set_title(f"{label}\n{subtitle}", fontsize=10)
        ax.set_ylabel("Rate / Mean")
        for bar, val in zip(bars, [c_val, v_val]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "guardrails_dashboard.png", dpi=150)
    plt.close(fig)
