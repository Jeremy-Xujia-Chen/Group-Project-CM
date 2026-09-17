"""End-to-end demo: calibrate, validate, run the policy scenarios, draw the map.

    python run_demo.py            # full run
    python run_demo.py --quick    # smaller and faster, for a live demonstration

Everything it writes lands in outputs/.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from abm import calibration, experiments, geography as geo, viz
from abm.model import MigrationModel, Params

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agents", type=int, default=40_000)
    p.add_argument("--years", type=int, default=20)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--quick", action="store_true", help="small run for a live demo")
    p.add_argument("--gif", action="store_true", help="write a GIF instead of MP4")
    p.add_argument("--no-calibrate", action="store_true", help="use the stored beta_home")
    p.add_argument("--states-csv", default=None, help="override the ACS state table")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.agents, args.years, args.seeds = 12_000, 12, 2

    OUT.mkdir(exist_ok=True)
    states = calibration.load_states(args.states_csv)
    params = Params(n_agents=args.agents, years=args.years)

    print(f"Loaded {len(states)} states from the ACS table.")

    # ---------------------------------------------------------------- calibrate
    target = calibration.observed_move_rate(states)
    print(f"\nObserved interstate migration rate, 18-34: {target:.3%} per year")

    if args.no_calibrate:
        print(f"Using stored beta_home = {params.beta_home}")
    else:
        print("Calibrating beta_home (stay-put bonus) to match that rate:")
        beta, achieved = calibration.calibrate_home_bias(
            target, params, states_csv=args.states_csv
        )
        params = replace(params, beta_home=beta)
        print(f"  -> beta_home = {beta:.3f}, simulated rate = {achieved:.3%}")

    # ---------------------------------------------------------------- baseline
    print("\nRunning the baseline scenario ...")
    baseline = MigrationModel(params, states_csv=args.states_csv)
    baseline_results = baseline.run()
    baseline_results.to_csv(OUT / "baseline_by_state_year.csv", index=False)

    merged = calibration.validate_against_census(baseline, states)
    scores = calibration.validation_scores(merged)
    print(
        f"Validation against ACS in-migration rates: "
        f"r = {scores['pearson_r']:.3f}, rank r = {scores['spearman_r']:.3f} "
        f"(n = {scores['n_states']})"
    )
    viz.plot_validation(merged, scores, OUT / "validation.png")
    merged.to_csv(OUT / "validation.csv", index=False)

    # ---------------------------------------------------------------- the map
    print("\nRendering the migration map ...")
    suffix = ".gif" if args.gif else ".mp4"
    anim_path = viz.animate_migration(
        baseline,
        OUT / f"migration{suffix}",
        subtitle=f"baseline, no new policy | {args.agents:,} agents | ACS-initialised",
    )
    viz.plot_snapshot(baseline, args.years, OUT / "migration_final_year.png")
    print(f"  {anim_path}")
    print(f"  {OUT / 'migration_final_year.png'}")

    # ---------------------------------------------------------------- scenarios
    print(f"\nRunning policy scenarios across {args.seeds} seeds ...")
    scenarios = experiments.make_scenarios(baseline.order, baseline.pop_18_34)
    results = experiments.run_scenarios(
        scenarios, params, seeds=list(range(args.seeds)), states_csv=args.states_csv
    )
    results.to_csv(OUT / "scenario_results.csv", index=False)

    measures = experiments.national_measures(results)
    measures.to_csv(OUT / "scenario_measures.csv", index=False)
    viz.plot_scenarios(measures, OUT / "scenarios.png")

    summary = experiments.scenario_summary(measures, final_year=args.years)
    summary.to_csv(OUT / "scenario_summary.csv", index=False)
    print("\nFinal-year national indicators by scenario:")
    cols = ["scenario", "talent_hhi_mean", "national_college_share_mean",
            "mean_rent_burden_mean", "income_sd_across_states_mean"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # ---------------------------------------------------------------- spillover
    treated = [s for s, v in scenarios["housing_concentrated"].housing.items() if v > 0]
    spill = experiments.spillover_to_neighbours(
        results, treated, geo.ADJACENCY, "housing_concentrated"
    )
    spill.to_csv(OUT / "spillover.csv", index=False)
    print(
        f"\nConcentrated housing investment in {', '.join(sorted(treated))} - "
        "final-year population vs baseline:"
    )
    print(spill.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

    print(f"\nDone. Everything written to {OUT}/")


if __name__ == "__main__":
    main()
