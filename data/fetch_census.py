"""Fetch state-level initial conditions for the ABM from the US Census ACS API.

Writes data/states_acs.csv - one row per state: population aged 18-34, earnings by
education, rents, degree share, employment, and the observed interstate in-migration
rate for the 18-34 cohort, which the model calibrates and validates against.

With --flows it also writes data/flows_acs.csv, the state-to-state migration matrix,
for validating individual corridors. That is optional and much slower.

Requires a free Census API key (https://api.census.gov/data/key_signup.html), supplied via
the CENSUS_API_KEY environment variable or a `.census_api_key` file in the project root.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

ACS_YEAR = 2023
FLOWS_YEAR = 2022  # flows dataset lags the detail tables
BASE = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
FLOWS_BASE = f"https://api.census.gov/data/{FLOWS_YEAR}/acs/flows"

# Census codes for the 50 states + DC. Puerto Rico (72) is excluded: it is outside the
# interstate migration system the model is about, and ACS flows treat it separately.
EXCLUDED_FIPS = {"72", "60", "66", "69", "78"}

# Age brackets covering 18-34, split by sex in table B01001.
AGE_18_34_MALE = [f"B01001_{n:03d}E" for n in range(7, 13)]
AGE_18_34_FEMALE = [f"B01001_{n:03d}E" for n in range(31, 37)]

# Table B07001 (geographic mobility by age) gives the observed interstate
# in-migration rate for exactly the 18-34 cohort, which the model calibrates to.
# 004-007 are the cohort totals; 068-071 are those who moved from another state.
MOBILITY_DENOM = [f"B07001_{n:03d}E" for n in range(4, 8)]
MOBILITY_INTERSTATE = [f"B07001_{n:03d}E" for n in range(68, 72)]

VARIABLES = {
    "NAME": "state_name",
    "B01003_001E": "pop_total",
    "B19013_001E": "median_hh_income",
    "B25064_001E": "median_gross_rent",       # monthly
    "B25071_001E": "median_rent_burden_pct",  # rent as % of household income
    "B23025_003E": "civilian_labor_force",
    "B23025_004E": "employed",
    "B23025_005E": "unemployed",
    "B15003_001E": "edu_pop_25plus",
    "B15003_022E": "edu_bachelors",
    "B15003_023E": "edu_masters",
    "B15003_024E": "edu_professional",
    "B15003_025E": "edu_doctorate",
    "B20004_001E": "earn_median_all",
    "B20004_003E": "earn_median_hs",
    "B20004_004E": "earn_median_somecollege",
    "B20004_005E": "earn_median_bachelors",
    "B20004_006E": "earn_median_graduate",
}


def get_api_key() -> str:
    key = os.environ.get("CENSUS_API_KEY", "").strip()
    if key:
        return key
    key_file = ROOT / ".census_api_key"
    if key_file.exists():
        key = key_file.read_text().strip()
        if key:
            return key
    sys.exit(
        "No Census API key found.\n"
        "  Get a free key at https://api.census.gov/data/key_signup.html\n"
        "  Then either:  export CENSUS_API_KEY=your_key\n"
        f"  or write it to: {key_file}"
    )


def _request(url: str, params: dict, attempts: int = 3) -> list[list[str]]:
    """GET a Census API endpoint that returns its rows as a JSON array-of-arrays."""
    for attempt in range(attempts):
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        if attempt == attempts - 1:
            raise RuntimeError(
                f"Census API returned {resp.status_code} for {url}\n{resp.text[:400]}"
            )
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def fetch_state_table(key: str) -> pd.DataFrame:
    codes = (
        list(VARIABLES)
        + AGE_18_34_MALE
        + AGE_18_34_FEMALE
        + MOBILITY_DENOM
        + MOBILITY_INTERSTATE
    )
    rows = _request(BASE, {"get": ",".join(codes), "for": "state:*", "key": key})

    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df[~df["state"].isin(EXCLUDED_FIPS)].copy()

    numeric = [c for c in df.columns if c != "NAME"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    # ACS uses large negative sentinels for suppressed/unavailable estimates.
    df[numeric] = df[numeric].mask(df[numeric] < -1e6)

    df["pop_18_34"] = df[AGE_18_34_MALE + AGE_18_34_FEMALE].sum(axis=1)
    df["bachelors_plus_share"] = (
        df[["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"]].sum(axis=1)
        / df["B15003_001E"]
    )
    df["unemployment_rate"] = df["B23025_005E"] / df["B23025_003E"]

    df["mobility_base_18_34"] = df[MOBILITY_DENOM].sum(axis=1)
    df["inmigrants_18_34"] = df[MOBILITY_INTERSTATE].sum(axis=1)
    df["interstate_inflow_rate_18_34"] = (
        df["inmigrants_18_34"] / df["mobility_base_18_34"]
    )

    df = df.rename(columns={**VARIABLES, "state": "state_fips"})
    keep = [
        "state_fips", "state_name", "pop_total", "pop_18_34",
        "median_hh_income", "median_gross_rent", "median_rent_burden_pct",
        "bachelors_plus_share", "unemployment_rate",
        "earn_median_all", "earn_median_hs", "earn_median_somecollege",
        "earn_median_bachelors", "earn_median_graduate",
        "mobility_base_18_34", "inmigrants_18_34", "interstate_inflow_rate_18_34",
    ]
    return df[keep].sort_values("state_fips").reset_index(drop=True)


def fetch_flows(key: str, state_fips: list[str]) -> pd.DataFrame:
    """Observed state-to-state migration flows, one row per origin-destination pair.

    The ACS flows API only publishes county-level geography, so state-to-state
    totals are built by summing county-to-county pairs that cross a state line.
    One request per state.
    """
    frames = []
    for fips in sorted(set(state_fips) - EXCLUDED_FIPS):
        rows = _request(
            FLOWS_BASE,
            {
                "get": "STATE2,MOVEDIN,MOVEDOUT,MOVEDNET",
                "for": "county:*",
                "in": f"state:{fips}",
                "key": key,
            },
        )
        part = pd.DataFrame(rows[1:], columns=rows[0])
        part["state_fips"] = fips
        frames.append(part)
        print(f"    {fips} ({len(part)} county pairs)")

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"STATE2": "other_fips"})
    df = df[df["other_fips"].notna()].copy()
    df["other_fips"] = df["other_fips"].astype(str).str.zfill(2)
    df = df[
        ~df["other_fips"].isin(EXCLUDED_FIPS) & (df["other_fips"] != df["state_fips"])
    ]

    for col in ("MOVEDIN", "MOVEDOUT", "MOVEDNET"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out = (
        df.groupby(["state_fips", "other_fips"], as_index=False)[
            ["MOVEDIN", "MOVEDOUT", "MOVEDNET"]
        ]
        .sum()
        .rename(
            columns={"MOVEDIN": "moved_in", "MOVEDOUT": "moved_out", "MOVEDNET": "moved_net"}
        )
    )
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flows",
        action="store_true",
        help="also fetch state-to-state flows (51 requests, a few minutes). "
        "Only needed to validate individual migration corridors; the model "
        "calibrates against the age-specific rates in the state table.",
    )
    args = parser.parse_args()

    key = get_api_key()
    DATA.mkdir(exist_ok=True)

    print(f"Fetching ACS {ACS_YEAR} 5-year state table ...")
    states = fetch_state_table(key)
    states.to_csv(DATA / "states_acs.csv", index=False)
    print(f"  wrote data/states_acs.csv  ({len(states)} states)")

    if args.flows:
        print(f"Fetching ACS {FLOWS_YEAR} flows, aggregated from counties ...")
        flows = fetch_flows(key, states["state_fips"].tolist())
        flows.to_csv(DATA / "flows_acs.csv", index=False)
        print(f"  wrote data/flows_acs.csv   ({len(flows)} state pairs)")


if __name__ == "__main__":
    main()
