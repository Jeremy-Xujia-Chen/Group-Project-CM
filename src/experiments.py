"""Experiment runners: policy builders, single-case runs, run summaries and cached result tables.

Each ``run_*_case`` function simulates one (scenario, seed) case and returns a
summary row. The ``*_runs`` loaders read the cached table from data/results
when it exists and otherwise simulate every case and write the table.
"""
import numpy as np
import pandas as pd

from src import parameters
from src.config import (
    CALIBRATION_SEEDS,
    CROSS_SEEDS,
    EXTENDED_INTENSITIES,
    EXTERNAL_ORIGIN_TARGET,
    INTERACTION_LEVELS,
    NETWORK_DECAYS,
    POLICY_TYPES,
    RESULTS_DIR,
    STANDARD_INTENSITY,
    SWEEP_INTERACTIONS,
    SWEEP_SEEDS,
)
from src.data import state_data
from src.extensions import DistanceDecayABM, OpenSystemABM
from src.model import PolicySocialABM
from src.parameters import PARAMS, policy_table
from utils.parallel import run_cases


def summarise_v7_run(model, history, target_state):
    mech = model.mechanism_history()
    target = history[history["state"] == target_state]
    system = history.groupby("year").first()
    accepted = mech["target_accepted"].sum()
    return {
        "target_inflow_rate": accepted / history.groupby("year")["population"].sum().sum(),
        "target_social_cost": mech["target_social_cost_sum"].sum() / max(mech["target_social_cost_n"].sum(), 1),
        "chain_share": mech["target_chain_moves"].sum() / max(accepted, 1),
        "housing_rejection": mech["target_housing_rejected"].sum() / max(mech["target_proposed"].sum(), 1),
        "job_rejection": target["job_rejection_rate_state"].mean(),
        "coorigin_clustering": system["coorigin_clustering"].iloc[-1],
    }


def lever_policy(target_state, lever, intensity):
    if lever == "combined":  # original V7 package: housing + employment, each carrying a 0.5x incentive
        return policy_table(target_state, housing=intensity, employment=intensity, migration_incentive=intensity)
    return policy_table(target_state, **{lever: intensity})  # pure single lever, no bundled incentive


def cross_policy(policy, intensity):
    # Section 9 design: each active single lever carries a 0.5 x intensity migration incentive.
    if policy == "baseline":
        return policy_table()
    h = intensity if policy in ("housing", "combined") else 0.0
    e = intensity if policy in ("employment", "combined") else 0.0
    return policy_table(parameters.FOCAL_STATE, housing=h, employment=e, migration_incentive=0.5 * intensity * ((h > 0) + (e > 0)))


def run_lever_case(target_state, lever, intensity, interaction, seed):
    # The model reports target-state metrics for parameters.FOCAL_STATE, so it is
    # pointed at target_state for this run and restored afterwards.
    saved_focal = parameters.FOCAL_STATE
    parameters.FOCAL_STATE = target_state
    try:
        model = PolicySocialABM(
            state_data, lever_policy(target_state, lever, intensity), seed=seed,
            interaction_strength=INTERACTION_LEVELS[interaction],
        )
        history, _, _ = model.run()
        summary = summarise_v7_run(model, history, target_state)
    finally:
        parameters.FOCAL_STATE = saved_focal
    return {"target_state": target_state, "lever": lever, "intensity": intensity,
            "interaction": interaction, "seed": seed, **summary}


def live_summary(target_state, policy_df, interaction, seed):
    """Re-simulate one case live (used by the reproducibility spot check)."""
    saved_focal, parameters.FOCAL_STATE = parameters.FOCAL_STATE, target_state
    try:
        model = PolicySocialABM(state_data, policy_df, seed=seed, interaction_strength=INTERACTION_LEVELS[interaction])
        history, _, _ = model.run()
        return summarise_v7_run(model, history, target_state)
    finally:
        parameters.FOCAL_STATE = saved_focal


def cached_or_run(filename, cases):
    path = RESULTS_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    runs = pd.DataFrame(run_cases(run_lever_case, cases))
    path.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(path, index=False)
    return runs


def run_decay_case(decay, beta_stay, policy, interaction, seed, recalibrated):
    params = dict(PARAMS, beta_current_state=beta_stay)
    pol = policy_table() if policy == "baseline" else policy_table(parameters.FOCAL_STATE, housing=1.0, employment=1.0, migration_incentive=1.0)
    kw = dict(seed=seed, interaction_strength=INTERACTION_LEVELS[interaction], params=params)
    model = PolicySocialABM(state_data, pol, **kw) if decay == 0 else DistanceDecayABM(state_data, pol, network_decay=decay, **kw)
    history, _, _ = model.run()
    s = model._state_indices()
    return {"network_decay": decay, "beta_current_state": beta_stay, "recalibrated": recalibrated, "policy": policy,
            "interaction": interaction, "seed": seed, "move_rate": history.groupby("year").first()["move_rate"].mean(),
            "same_state_contact_share": float(np.mean(s[model.network] == s[:, None])),
            **summarise_v7_run(model, history, parameters.FOCAL_STATE)}


def network_decay_runs(path=RESULTS_DIR / "network_decay_runs.csv"):
    """Section 13.4 runs: each decay level with the original and a re-calibrated beta_current_state."""
    if path.exists():
        return pd.read_csv(path)

    def mean_move_rate(decay, beta):
        return np.mean([run_decay_case(decay, beta, "baseline", "medium", s, False)["move_rate"] for s in CALIBRATION_SEEDS])
    target_rate = mean_move_rate(0.0, PARAMS["beta_current_state"])
    betas = {0.0: PARAMS["beta_current_state"]}
    for decay in NETWORK_DECAYS[1:]:
        lo, hi = 1.5, PARAMS["beta_current_state"]
        for _ in range(9):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mean_move_rate(decay, mid) > target_rate else (lo, mid)
        betas[decay] = (lo + hi) / 2
    cases = [(d, b, p, i, s, recal) for recal in [False, True] for d in NETWORK_DECAYS
             for b in [betas[d] if recal else PARAMS["beta_current_state"]]
             for p in ["baseline", "combined"] for i in SWEEP_INTERACTIONS for s in CROSS_SEEDS]
    decay_runs = pd.DataFrame(run_cases(run_decay_case, cases))
    decay_runs.to_csv(path, index=False)
    return decay_runs


def run_open_case(mode, rate, design, policy, intensity, interaction, seed):
    model = OpenSystemABM(state_data, cross_policy(policy, intensity) if intensity > 0 else policy_table(), seed=seed,
                          interaction_strength=INTERACTION_LEVELS[interaction], external_rate=rate, capacity_mode=mode)
    history, _, _ = model.run()
    mech, ext = model.mechanism_history(), pd.DataFrame(model.external_rows)
    agent_years = history.groupby("year")["population"].sum().sum()
    internal_movers = (history.groupby("year").first()["move_rate"] * history.groupby("year")["population"].sum()).sum()
    accepted = mech["target_accepted"].sum()
    return {"capacity_mode": mode, "external_rate": rate, "design": design, "policy": policy, "intensity": intensity,
            "interaction": interaction, "seed": seed,
            "target_inflow_rate": accepted / agent_years,
            "target_total_inflow_rate": (accepted + ext["target_external_accepted"].sum()) / agent_years,
            "chain_share": mech["target_chain_moves"].sum() / max(accepted, 1),
            "housing_rejection": (mech["target_housing_rejected"].sum() + ext["target_external_rejected"].sum())
                                 / max(mech["target_proposed"].sum() + ext["target_external_proposed"].sum(), 1),
            "external_share_system": ext["external_accepted"].sum() / max(ext["external_accepted"].sum() + internal_movers, 1)}


def open_system_runs(path=RESULTS_DIR / "open_system_runs.csv"):
    """Section 13.5 runs: external inflow rate calibrated by bisection per capacity mode, then cross and sweep designs."""
    if path.exists():
        return pd.read_csv(path)

    def external_share(mode, rate):
        return np.mean([run_open_case(mode, rate, "cal", "baseline", 0.0, "medium", s)["external_share_system"] for s in CALIBRATION_SEEDS])
    cases = []
    for mode in ["shared", "scaled"]:
        lo, hi = 0.03, 0.25
        for _ in range(8):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if external_share(mode, mid) < EXTERNAL_ORIGIN_TARGET else (lo, mid)
        rate = (lo + hi) / 2
        cases += [(mode, rate, "cross", p, STANDARD_INTENSITY, i, s) for p in POLICY_TYPES for i in SWEEP_INTERACTIONS for s in CROSS_SEEDS]
        cases += [(mode, rate, "sweep", "combined" if x > 0 else "baseline", x, "medium", s) for x in EXTENDED_INTENSITIES for s in SWEEP_SEEDS]
    open_runs = pd.DataFrame(run_cases(run_open_case, cases))
    open_runs.to_csv(path, index=False)
    return open_runs
