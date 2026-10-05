"""Project-wide constants: file locations, state set, experimental design and seeds."""
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"        # empirical inputs (ACS, IRS, BRFSS, state snapshot)
RESULTS_DIR = DATA_DIR / "results"     # cached seed-level simulation outputs

RANDOM_SEED = 42
BASE_YEAR = 2026

STATE_CODES = ["CA", "TX", "NY", "FL", "MA", "NC", "IL", "WA"]
STATE_NAMES = {
    "CA": "California", "TX": "Texas", "NY": "New York", "FL": "Florida",
    "MA": "Massachusetts", "NC": "North Carolina", "IL": "Illinois", "WA": "Washington",
}
MODEL_STATE_FIPS = {"CA": 6, "TX": 48, "NY": 36, "FL": 12, "MA": 25, "NC": 37, "IL": 17, "WA": 53}

# Full research design. Reduce these values only for debugging.
BASELINE_AGENTS = 1000
BASELINE_YEARS = 20
BASELINE_SEEDS = list(range(101, 109))

EXPERIMENT_AGENTS = 600
EXPERIMENT_YEARS = 12
CROSS_SEEDS = list(range(201, 209))
SWEEP_SEEDS = list(range(301, 309))
SHOCK_SEEDS = list(range(401, 407))
SENSITIVITY_SEEDS = list(range(501, 505))
CALIBRATION_SEEDS = (901, 902, 903, 904, 905)

N_JOBS = 1  # deterministic final execution; avoids process-backend stalls in notebook kernels
N_AGENTS = BASELINE_AGENTS
N_YEARS = BASELINE_YEARS

POLICY_COLS = [
    "housing_policy",
    "employment_incentive",
    "migration_incentive",
]

SOCIAL_UTILITY_WEIGHTS = {
    "peer": 0.30,
    "coorigin": 0.20,
    "home": 0.10,
    "employment": 0.20,
    "income": 0.10,
    "affordability": 0.10,
}

# Experiment factors (Sections 9-13).
INTERACTION_LEVELS = {"weak": 0.60, "medium": 1.00, "strong": 1.40}
POLICY_TYPES = ["baseline", "housing", "employment", "combined"]
STANDARD_INTENSITY = 1.0

# 0.5-unit grid to 4.0 (comparable with Section 10) plus 5 and 6 to detect
# saturation beyond the original range; all three social-tie levels.
EXTENDED_INTENSITIES = [float(x) for x in np.arange(0.0, 4.01, 0.5)] + [5.0, 6.0]
SWEEP_INTERACTIONS = ["weak", "medium", "strong"]
LEVERS = ["housing", "employment", "migration_incentive", "combined"]

SHOCK_NAMES = ["baseline", "financial_crisis", "pandemic", "labor_shortage"]
NETWORK_DECAYS = [0.0, 2.0, 5.0, 10.0]
EXTERNAL_ORIGIN_TARGET = 0.62  # ACS PUMS share of in-movers arriving from other US states (Section 3.1)
