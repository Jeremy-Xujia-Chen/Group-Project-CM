"""Calibrate the model's inertia parameter to Census data, and validate against it.

The model has one parameter with no direct empirical counterpart: `beta_home`, the
utility bonus for staying put. Everything else is either observed (wages, rents,
degree shares) or a structural elasticity. So `beta_home` is pinned down by making
the model reproduce the *aggregate* interstate migration rate of 18-34 year olds
that the ACS observes. The model's spatial pattern - which states gain and which
lose - is then a prediction, not an input, and can be checked against the data.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .model import MigrationModel, Params

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_states(states_csv: Path | str | None = None) -> pd.DataFrame:
    path = Path(states_csv) if states_csv else DATA / "states_acs.csv"
    df = pd.read_csv(path, dtype={"state_fips": str})
    df["state_fips"] = df["state_fips"].str.zfill(2)
    return df


def observed_move_rate(states: pd.DataFrame) -> float:
    """National share of 18-34 year olds who lived in a different state a year ago."""
    return float(
        states["inmigrants_18_34"].sum() / states["mobility_base_18_34"].sum()
    )


def simulated_move_rate(model: MigrationModel, first: int = 1, last: int = 5) -> float:
    """Mean annual share of agents who changed state, over the given year range."""
    rates = []
    for snap in model.history:
        if first <= snap["year"] <= last:
            rates.append(snap["outflow"].sum() / snap["population"].sum())
    return float(np.mean(rates)) if rates else 0.0


def calibrate_home_bias(
    target_rate: float,
    params: Params | None = None,
    states_csv: Path | str | None = None,
    bracket: tuple[float, float] = (1.0, 12.0),
    tolerance: float = 2e-4,
    max_iter: int = 18,
    verbose: bool = True,
) -> tuple[float, float]:
    """Bisect on `beta_home` until the simulated move rate matches `target_rate`.

    Returns (beta_home, achieved_rate). The move rate falls monotonically as the
    stay-put bonus rises, so bisection is well behaved.
    """
    base = params or Params()
    # A short run is enough: the move rate is set by the choice model, not by the
    # slow macro feedback, so it stabilises within a few years.
    base = replace(base, years=6)

    lo, hi = bracket

    def rate_at(beta: float) -> float:
        m = MigrationModel(replace(base, beta_home=beta), states_csv=states_csv)
        m.run()
        return simulated_move_rate(m)

    rate_lo, rate_hi = rate_at(lo), rate_at(hi)
    if not (rate_hi <= target_rate <= rate_lo):
        raise ValueError(
            f"Target rate {target_rate:.4f} outside achievable range "
            f"[{rate_hi:.4f}, {rate_lo:.4f}] for beta_home in {bracket}."
        )

    best = (lo, rate_lo)
    for i in range(max_iter):
        mid = 0.5 * (lo + hi)
        rate = rate_at(mid)
        if verbose:
            print(f"  iter {i:2d}  beta_home={mid:6.3f}  move rate={rate:7.4f}")
        best = (mid, rate)
        if abs(rate - target_rate) < tolerance:
            break
        if rate > target_rate:
            lo = mid
        else:
            hi = mid
    return best


def validate_against_census(
    model: MigrationModel,
    states: pd.DataFrame,
    year: int = 1,
) -> pd.DataFrame:
    """Compare simulated per-state in-migration rates against the observed ones."""
    snap = model.history[year]
    sim = pd.DataFrame(
        {
            "abbr": snap["abbr"],
            "sim_inflow": snap["inflow"],
            "sim_population": snap["population"],
        }
    )
    sim["sim_inflow_rate"] = sim["sim_inflow"] / sim["sim_population"].clip(lower=1)

    obs = states[["state_fips", "interstate_inflow_rate_18_34", "inmigrants_18_34"]].copy()
    from . import geography as geo

    obs["abbr"] = obs["state_fips"].map(geo.fips_to_abbr())
    merged = sim.merge(obs.dropna(subset=["abbr"]), on="abbr", how="inner")
    return merged.rename(columns={"interstate_inflow_rate_18_34": "obs_inflow_rate"})


def validation_scores(merged: pd.DataFrame) -> dict[str, float]:
    a = merged["sim_inflow_rate"].to_numpy()
    b = merged["obs_inflow_rate"].to_numpy()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    rank_a = pd.Series(a).rank().to_numpy()
    rank_b = pd.Series(b).rank().to_numpy()
    return {
        "pearson_r": float(np.corrcoef(a, b)[0, 1]),
        "spearman_r": float(np.corrcoef(rank_a, rank_b)[0, 1]),
        "n_states": int(mask.sum()),
    }
