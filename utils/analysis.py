"""Small analysis helpers used by the notebook: cost scaling, net-benefit index and cache checks."""
import numpy as np

TULSA_DOLLAR_PER_INTENSITY = 10_000
BARTIK_BUT_FOR_LOW = 0.58
BARTIK_BUT_FOR_HIGH = 0.70


def intensity_to_dollar_equivalent(intensity, dollar_per_unit=TULSA_DOLLAR_PER_INTENSITY):
    """Reference scale only; not a fiscal or causal calibration for general young adults."""
    return np.asarray(intensity, dtype=float) * float(dollar_per_unit)


def minmax(s):
    """Min-max scale a series to [0, 1] (all zeros if constant)."""
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0.0


def net_benefit(df, w_s, w_h):
    return df["inflow_scaled"] - w_s * df["social_cost_scaled"] - w_h * df["housing_rejection_scaled"]


def seed_column(df):
    """Name of the replicate/seed column in a run table, or None."""
    for c in ["seed", "run_seed", "replicate", "rep"]:
        if c in df.columns:
            return c
    return None


def compare(label, live, cached_row, aliases):
    """Assert that a live re-simulation matches a cached row to 1e-9 on every shared metric."""
    checked = []
    for key, names in aliases.items():
        name = next((n for n in names if n in cached_row.index), None)
        if name is None:
            continue
        diff = abs(float(live[key]) - float(cached_row[name]))
        assert diff < 1e-9, f"{label}: {key} live={live[key]:.6f} cached={cached_row[name]:.6f}"
        checked.append(key)
    assert "target_inflow_rate" in checked, f"{label}: cached row has no target_inflow_rate"
    return {"case": label, "metrics_checked": len(checked), "target_inflow_rate": live["target_inflow_rate"], "status": "match"}
