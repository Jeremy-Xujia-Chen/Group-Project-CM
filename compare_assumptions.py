"""Compare three assumptions about what distance means in the migration decision.

Each variant is calibrated separately to the same observed national move rate, so
they differ only in *where* people go, not how many move. Whether any of them
reproduces the observed spatial pattern is the question.

    python compare_assumptions.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from abm import calibration, geography as geo, viz
from abm.model import MigrationModel, Params

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SEEDS = list(range(6))

MODES = {
    "centroid": "Centroid distance only",
    "border": "Distance beyond leaving your own state",
    "size_home": "Stay-put bonus scaled by state size",
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    states = calibration.load_states()
    target = calibration.observed_move_rate(states)
    print(f"Observed interstate move rate, 18-34: {target:.3%}\n")

    radii = geo.state_radii()
    print("State radii (km): " + ", ".join(
        f"{a}={radii[a]:.0f}" for a in ("RI", "DC", "DE", "CA", "TX", "AK")
    ))

    rows, per_state = [], []
    for mode, label in MODES.items():
        print(f"\n--- {mode}: {label} ---")
        base = Params(n_agents=40_000, years=20, geography_mode=mode)
        beta, achieved = calibration.calibrate_home_bias(
            target, base, verbose=False
        )
        print(f"  calibrated beta_home = {beta:.3f} (move rate {achieved:.3%})")

        for seed in SEEDS:
            model = MigrationModel(replace(base, beta_home=beta, seed=seed))
            model.run()

            merged = calibration.validate_against_census(model, states)
            scores = calibration.validation_scores(merged)

            # DC is a single extreme point that can carry the whole correlation,
            # so the same statistic is reported with it removed.
            no_dc = merged[merged["abbr"] != "DC"]
            r_no_dc = float(
                np.corrcoef(no_dc["obs_inflow_rate"], no_dc["sim_inflow_rate"])[0, 1]
            )

            # The gradient the observed data shows most strongly: smaller states
            # have higher interstate turnover.
            grad_sim = float(
                np.corrcoef(np.log(model.pop_18_34), merged["sim_inflow_rate"])[0, 1]
            )

            rows.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "beta_home": beta,
                    "move_rate": achieved,
                    "pearson_r": scores["pearson_r"],
                    "pearson_r_no_dc": r_no_dc,
                    "spearman_r": scores["spearman_r"],
                    "size_gradient_sim": grad_sim,
                }
            )
            per_state.append(merged.assign(mode=mode, seed=seed))
            if seed == SEEDS[0]:
                viz.plot_validation(merged, scores, OUT / f"validation_{mode}.png")

        got = pd.DataFrame([r for r in rows if r["mode"] == mode])
        for col in ("pearson_r", "pearson_r_no_dc", "spearman_r", "size_gradient_sim"):
            print(f"  {col:20s} {got[col].mean(): .3f} +/- {got[col].std():.3f}")

    summary = pd.DataFrame(rows)

    obs_grad = float(
        np.corrcoef(
            np.log(states["pop_18_34"]), states["interstate_inflow_rate_18_34"]
        )[0, 1]
    )
    summary["size_gradient_observed"] = obs_grad

    summary.to_csv(OUT / "assumption_comparison.csv", index=False)
    pd.concat(per_state, ignore_index=True).to_csv(
        OUT / "assumption_comparison_by_state.csv", index=False
    )

    stats = (
        summary.groupby("mode")[
            ["pearson_r", "pearson_r_no_dc", "spearman_r", "size_gradient_sim"]
        ]
        .agg(["mean", "std"])
        .round(3)
    )

    print("\n" + "=" * 78)
    print(
        f"Distance assumptions, {len(SEEDS)} seeds each, all calibrated to the "
        "same move rate"
    )
    print("=" * 78)
    print(stats.to_string())
    print(f"\nObserved size gradient, corr(log population, inflow rate): {obs_grad:.3f}")
    print("A variant reproduces the real gradient by getting close to that value.")
    print(f"\nWritten to {OUT}/assumption_comparison.csv")


if __name__ == "__main__":
    main()
