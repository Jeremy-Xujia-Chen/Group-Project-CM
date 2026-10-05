"""Empirical inputs: state snapshot, state geography and IRS interstate routes (read from data/raw)."""
import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR, STATE_CODES
from utils.stats import haversine_km, mean_index


def build_real_state_snapshot(path=RAW_DATA_DIR / "state_inputs.csv"):
    raw = pd.read_csv(path, index_col="state").reindex(STATE_CODES)

    raw["employment_rate_labor_force"] = (
        raw["employed"] / raw["civilian_labor_force"]
    )
    raw["vacancy_rate"] = (
        raw["vacant_housing_units"] / raw["housing_units"]
    )
    raw["ipeds_enrollment_per_young_adult"] = (
        raw["ipeds_12mo_enrollment"]
        / raw["young_adult_population_est_18_35"]
    )
    return raw


def prepare_state_data(raw):
    df = raw.copy()

    df["youth_pop_weight"] = mean_index(
        df["young_adult_population_est_18_35"]
    )
    df["wage_index"] = mean_index(df["median_household_income"])
    df["rent_index"] = mean_index(df["median_gross_rent_monthly"])
    df["job_index"] = mean_index(df["employment_rate_labor_force"])
    df["education_access_index"] = mean_index(
        df["ipeds_enrollment_per_young_adult"]
    )
    df["tuition_index"] = mean_index(df["public_4yr_instate_tuition"])
    df["childcare_cost_index"] = mean_index(
        df["annual_infant_childcare_price"]
    )
    df["housing_capacity_index"] = mean_index(df["vacancy_rate"])
    df["college_share_proxy"] = df["bachelor_plus_share_25plus"]

    return df


def load_state_coords(path=RAW_DATA_DIR / "state_coords.csv"):
    coords = pd.read_csv(path)
    return {row.state: (row.lat, row.lon) for row in coords.itertuples()}


def build_distance_matrix(coords):
    distance_km = pd.DataFrame(
        index=STATE_CODES, columns=STATE_CODES, dtype=float
    )

    for s1 in STATE_CODES:
        for s2 in STATE_CODES:
            distance_km.loc[s1, s2] = haversine_km(
                *coords[s1], *coords[s2]
            )
    return distance_km


def load_irs_flows(path=RAW_DATA_DIR / "irs_routes.csv"):
    irs_flows = pd.read_csv(path)
    irs_flows["observed_share"] = (
        irs_flows["returns"]
        / irs_flows.groupby("origin")["returns"].transform("sum")
    )
    return irs_flows


def build_irs_flow_matrix(irs_flows):
    irs_flow_matrix = (
        irs_flows.pivot(
            index="origin", columns="destination", values="returns"
        )
        .reindex(index=STATE_CODES, columns=STATE_CODES)
    )

    for s in STATE_CODES:
        irs_flow_matrix.loc[s, s] = np.nan
    return irs_flow_matrix


real_raw_data = build_real_state_snapshot()
state_data = prepare_state_data(real_raw_data)

STATE_COORDS = load_state_coords()
distance_km = build_distance_matrix(STATE_COORDS)
distance_index = distance_km / distance_km.to_numpy().max()

irs_flows = load_irs_flows()
irs_flow_matrix = build_irs_flow_matrix(irs_flows)
