"""Policy scenarios and the aggregate measures used to compare them.

Every scenario spends the same total budget, expressed as intensity-units summed
across states, so differences in outcome reflect *where and how* money is spent
rather than how much.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .model import MigrationModel, Params, Policy

BUDGET = 8.0  # total intensity-units available to the whole country


def make_scenarios(
    order: list[str],
    populations: np.ndarray,
    budget: float = BUDGET,
    n_concentrated: int = 8,
) -> dict[str, Policy]:
    """Build the comparison set. All scenarios cost `budget` except the baseline."""
    even = budget / len(order)
    spread = {s: even for s in order}

    # The concentrated scenarios put the whole budget into the largest states, at
    # the maximum per-state intensity of 1.0.
    biggest = [order[i] for i in np.argsort(populations)[::-1][:n_concentrated]]
    focused = {s: budget / n_concentrated for s in biggest}

    return {
        "baseline": Policy(),
        "education": Policy(education=dict(spread)),
        "housing": Policy(housing=dict(spread)),
        "employment": Policy(employment=dict(spread)),
        "mixed": Policy(
            education={s: even / 3 for s in order},
            housing={s: even / 3 for s in order},
            employment={s: even / 3 for s in order},
        ),
        "housing_concentrated": Policy(housing=dict(focused)),
    }


def run_scenarios(
    scenarios: dict[str, Policy],
    params: Params,
    seeds: list[int],
    states_csv=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every scenario across every seed. Returns one long tidy frame."""
    frames = []
    for name, policy in scenarios.items():
        for seed in seeds:
            model = MigrationModel(replace(params, seed=seed), policy, states_csv=states_csv)
            df = model.run()
            df["scenario"] = name
            df["seed"] = seed
            frames.append(df)
        if verbose:
            print(f"  ran {name:22s} ({len(seeds)} seeds)")
    return pd.concat(frames, ignore_index=True)


def national_measures(results: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-state output into the national indicators we compare."""

    def agg(g: pd.DataFrame) -> pd.Series:
        pop = g["population"].to_numpy()
        share = pop / pop.sum()
        college = g["college_share"].to_numpy() * pop
        college_share_by_state = g["college_share"].to_numpy()
        return pd.Series(
            {
                # Herfindahl index: how concentrated the graduate pool is by state.
                "talent_hhi": float(
                    ((college / max(college.sum(), 1.0)) ** 2).sum()
                ),
                "population_hhi": float((share ** 2).sum()),
                "national_college_share": float(college.sum() / pop.sum()),
                "college_share_sd": float(college_share_by_state.std()),
                "mean_rent_burden": float(
                    (g["mean_rent_burden"].to_numpy() * share).sum()
                ),
                "mean_income": float((g["mean_income"].to_numpy() * share).sum()),
                "income_sd_across_states": float(g["mean_income"].std()),
                "churn": float(g["outflow"].sum()),
            }
        )

    return (
        results.groupby(["scenario", "seed", "year"], as_index=False)
        .apply(agg, include_groups=False)
        .reset_index(drop=True)
    )


def scenario_summary(measures: pd.DataFrame, final_year: int) -> pd.DataFrame:
    """Mean and spread of each indicator in the final year, by scenario."""
    final = measures[measures["year"] == final_year]
    cols = [
        "talent_hhi", "population_hhi", "national_college_share",
        "college_share_sd", "mean_rent_burden", "mean_income",
        "income_sd_across_states",
    ]
    out = final.groupby("scenario")[cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def spillover_to_neighbours(
    results: pd.DataFrame,
    treated: list[str],
    adjacency: dict[str, list[str]],
    scenario: str,
    baseline: str = "baseline",
) -> pd.DataFrame:
    """Did a state's policy cost its neighbours population, relative to baseline?

    Compares final-year population in three groups: the treated states, the states
    bordering them, and everywhere else.
    """
    neighbours = {n for t in treated for n in adjacency.get(t, [])} - set(treated)

    def group_of(abbr: str) -> str:
        if abbr in treated:
            return "treated"
        if abbr in neighbours:
            return "neighbour"
        return "other"

    final_year = results["year"].max()
    sub = results[
        results["scenario"].isin([scenario, baseline]) & (results["year"] == final_year)
    ].copy()
    sub["group"] = sub["abbr"].map(group_of)

    pivot = (
        sub.groupby(["scenario", "group", "seed"], as_index=False)["population"]
        .sum()
        .pivot_table(index=["group", "seed"], columns="scenario", values="population")
        .reset_index()
    )
    pivot["pct_change"] = 100 * (pivot[scenario] / pivot[baseline] - 1.0)
    return pivot.groupby("group", as_index=False)["pct_change"].agg(["mean", "std"])
