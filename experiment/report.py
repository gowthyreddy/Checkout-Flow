def print_report(freq: dict, bayes: dict, guardrails: dict, power: dict):
    flags = [k for k, v in guardrails.items() if v["status"] == "FLAG"]
    go = bayes["prob_variant_wins"] >= 0.95 and freq["significant"] and len(flags) == 0
    recommendation = "GO" if go else ("MONITOR" if bayes["prob_variant_wins"] >= 0.85 else "NO-GO")

    sep = "=" * 62
    print(f"\n{sep}")
    print("  EXPERIMENT BRIEF -- CHECKOUT FLOW A/B TEST")
    print(sep)
    print(f"  Dataset              : Online Shoppers Purchasing Intention")
    print(f"  Randomization unit   : Session (visitor-level)")
    print(f"  Primary metric       : Purchase conversion (Revenue = True)")
    print(f"  Novelty effect window: Exclude first 2 weeks post-launch")
    print(f"  Pre-registered alpha : {power['alpha']}  |  Power: {power['power']}")
    print(f"  Required n/variant   : {power['n_required_per_variant']:,}")
    print(sep)
    print("  FREQUENTIST RESULTS")
    print(f"    Control rate       : {freq['p_control']:.4%}")
    print(f"    Variant rate       : {freq['p_variant']:.4%}")
    print(f"    Absolute lift      : {freq['absolute_lift']:+.4%}")
    print(f"    Relative lift      : {freq['relative_lift_pct']:+.2f}%")
    print(f"    z-stat             : {freq['z_stat']}")
    print(f"    p-value            : {freq['p_value']}")
    print(f"    95% CI on diff     : ({freq['ci_95'][0]:.4%}, {freq['ci_95'][1]:.4%})")
    print(f"    Significant        : {'YES' if freq['significant'] else 'NO'}")
    print(sep)
    print("  BAYESIAN RESULTS")
    print(f"    P(variant > ctrl)  : {bayes['prob_variant_wins']:.1%}")
    print(f"    Expected loss      : {bayes['expected_loss']:.6f}  (threshold: 0.0005)")
    print(f"    Safe to stop       : {'YES' if bayes['safe_to_stop'] else 'NO'}")
    print(f"    95% HDI control    : ({bayes['hdi_95_control'][0]:.4%}, {bayes['hdi_95_control'][1]:.4%})")
    print(f"    95% HDI variant    : ({bayes['hdi_95_variant'][0]:.4%}, {bayes['hdi_95_variant'][1]:.4%})")
    print(sep)
    print("  GUARDRAIL METRICS")
    labels = {
        "session_duration": "Session Duration",
        "bounce_rate":      "Bounce Rate",
        "page_value":       "Page Value",
    }
    for metric, result in guardrails.items():
        status = result["status"]
        marker = "+" if status == "PASS" else "!"
        change = result.get("change_pct", 0)
        print(f"    [{marker}] {labels.get(metric, metric):<25} {status}  "
              f"(change={change:+.2f}%  p={result['p_value']})")
    if flags:
        print(f"\n    [!] Flagged: {', '.join(flags)} -- monitor post-launch")
    print(sep)
    print(f"  RECOMMENDATION: {recommendation}")
    if recommendation == "GO":
        print("    Both frameworks confirm significance. No guardrail violations.")
    elif recommendation == "MONITOR":
        print("    Bayesian signal is strong but proceed with post-launch monitoring.")
        if flags:
            print(f"    Watch: {', '.join(flags)}")
    else:
        print("    Insufficient evidence or guardrail violations block launch.")
    print(sep + "\n")
