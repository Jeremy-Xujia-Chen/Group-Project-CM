"""Model parameters, shock scenarios, policy tables and the focal policy state (Section 4)."""
import numpy as np
import pandas as pd

from src.calibration import CALIBRATED_BETA
from src.config import POLICY_COLS, STATE_CODES
from src.data import state_data

# Age-specific mortality pattern for the 18–35 study population.
MORTALITY_AGE = np.arange(18, 36, dtype=float)
MORTALITY_QX = np.array([
    0.000731, 0.000837, 0.000949, 0.001065, 0.001170, 0.001259,
    0.001335, 0.001406, 0.001480, 0.001560, 0.001651, 0.001749,
    0.001849, 0.001947, 0.002040, 0.002128, 0.002216, 0.002308,
])


def mortality_probability(age):
    return np.interp(
        np.asarray(age, dtype=float),
        MORTALITY_AGE,
        MORTALITY_QX,
        left=MORTALITY_QX[0],
        right=MORTALITY_QX[-1],
    )


PARAMS = {
    # IRS-calibrated destination attractiveness
    "beta_income": float(CALIBRATED_BETA[0]),
    "beta_jobs": float(CALIBRATED_BETA[1]),
    "beta_rent": float(CALIBRATED_BETA[2]),
    "beta_distance": float(CALIBRATED_BETA[3]),
    "beta_education": float(CALIBRATED_BETA[4]),
    "beta_childcare": float(CALIBRATED_BETA[5]),

    # Stay/move and social interaction
    "beta_current_state": 3.20,
    "beta_home_state": 0.20,
    "migration_cost": 1.00,
    "temperature": 0.90,
    "beta_unemployment_push": 0.45,
    "beta_job_rejection_push": 0.75,
    "beta_peer_network": 1.00,
    "beta_coorigin": 0.60,
    "beta_social_dissatisfaction": 0.90,
    "social_satisfaction_threshold": 0.35,
    "beta_crowding": 0.00,  # optional; activated only in sensitivity analysis
    "network_k": 8,
    "annual_pair_formation_rate": 0.08,

    # Direct policy pull + capacity effects
    "beta_migration_incentive": 0.55,
    "beta_housing_signal": 0.30,
    "beta_employment_signal": 0.40,
    "housing_entry_factor": 0.80,
    "housing_policy_slots": 0.05,
    "employment_policy_slots": 0.07,
    "housing_policy_effect": 0.030,
    "employment_policy_growth": 0.020,
    # Lagged rent response to excess housing demand. 0.15 is the central value;
    # Section 13 scans 0.00/0.10/0.15/0.20 as a robustness check.
    "rent_rejection_elasticity": 0.15,
    # Smooth housing-market rationing: availability becomes probabilistic as
    # remaining capacity approaches zero, avoiding a discontinuous hard wall.
    "housing_market_softness": 0.10,
    "housing_capacity_buffer": 0.05,

    # Demography
    "international_entry_rate": 0.008,
    "birth_peak_probability": 0.065,
    "birth_age_center": 29.0,
    "birth_age_spread": 5.0,
    "max_dependent_children": 3,
}

# Shock dictionaries use model-year indices. Recovery is linear after the active period.
SHOCK_SCENARIOS = {
    "baseline": {},
    "financial_crisis": {
        "start": 5, "duration": 2, "recovery_years": 3,
        "job_slot_multiplier": 0.72,
        "wage_multiplier": 0.92,
        "mortality_multiplier": 1.0,
        "migration_cost_multiplier": 1.0,
    },
    "pandemic": {
        "start": 5, "duration": 2, "recovery_years": 2,
        "job_slot_multiplier": 0.78,
        "wage_multiplier": 0.95,
        "mortality_multiplier": 2.50,
        "migration_cost_multiplier": 1.25,
    },
    "labor_shortage": {
        "start": 5, "duration": 4, "recovery_years": 2,
        "job_slot_multiplier": 1.25,
        "wage_multiplier": 1.06,
        "mortality_multiplier": 1.0,
        "migration_cost_multiplier": 1.0,
    },
}


def policy_table(
    target_state=None,
    housing=0.0,
    employment=0.0,
    migration_incentive=0.0,
    state_incentives=None,
):
    p = pd.DataFrame(0.0, index=STATE_CODES, columns=POLICY_COLS)
    if target_state is not None:
        p.loc[target_state, "housing_policy"] = float(housing)
        p.loc[target_state, "employment_incentive"] = float(employment)
        p.loc[target_state, "migration_incentive"] = float(migration_incentive)
    if state_incentives is not None:
        for state, value in state_incentives.items():
            p.loc[state, "migration_incentive"] = float(value)
    return p


# Focal state is chosen mechanically, not because it produces a preferred result.
opportunity_index = (
    state_data["job_index"]
    + state_data["housing_capacity_index"]
    - state_data["rent_index"]
)
FOCAL_STATE = (
    (opportunity_index - opportunity_index.median())
    .abs()
    .sort_values()
    .index[0]
)
