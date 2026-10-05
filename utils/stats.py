"""Statistical helpers: confidence intervals, index scaling, power, saturation fits and thresholds."""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t as student_t


def mean_t_ci(series):
    """Mean and 95% Student-t half-width of a series (NaNs dropped)."""
    x = np.asarray(pd.Series(series).dropna(), dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    mean = float(x.mean())
    if len(x) == 1:
        return mean, np.nan
    half = float(
        student_t.ppf(0.975, len(x) - 1)
        * x.std(ddof=1)
        / np.sqrt(len(x))
    )
    return mean, half


def logistic(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def mean_index(series):
    """Scale a series so its mean is 1."""
    x = series.astype(float)
    return x / x.mean()


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = (
        np.sin(dp / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


def power_summary(effects, n_seeds, alpha=0.05, power=0.80, max_seeds=2000):
    """Minimum detectable effect at the given power, and seeds needed for each observed effect."""
    df = n_seeds - 1
    se = effects["ci95_half_width"] / student_t.ppf(1 - alpha / 2, df)
    sd = se * np.sqrt(n_seeds)
    out = effects[["policy", "metric", "strong_minus_weak_policy_effect", "ci95_half_width"]].copy()
    out["mde_80pct_power"] = (student_t.ppf(1 - alpha / 2, df) + student_t.ppf(power, df)) * se
    def seeds_needed(effect, s):
        for n in range(3, max_seeds + 1):
            if (student_t.ppf(1 - alpha / 2, n - 1) + student_t.ppf(power, n - 1)) * s / np.sqrt(n) <= abs(effect):
                return n
        return np.nan
    out["seeds_for_observed_effect"] = [seeds_needed(e, s) for e, s in zip(out["strong_minus_weak_policy_effect"], sd)]
    return out


def hill_with_baseline(x, y0, A, K, n):
    """Hill saturation curve y0 + A x^n / (K^n + x^n)."""
    x = np.asarray(x, dtype=float)
    return y0 + A * x**n / (K**n + x**n)


def fit_hill(x, y):
    """Least-squares Hill fit; returns (y0, A, K, n)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    y0_guess = float(y[np.argmin(x)])
    A_guess = max(float(y.max() - y0_guess), 1e-6)
    p0 = [y0_guess, A_guess, float(np.median(x[x > 0])), 2.0]
    bounds = ([-np.inf, 0.0, 1e-3, 0.3], [np.inf, 10 * max(abs(A_guess), 1e-3) + 1.0, 50.0, 8.0])
    popt, _ = curve_fit(hill_with_baseline, x, y, p0=p0, bounds=bounds, maxfev=20000)
    return popt


def thresholds_by_seed(runs, group_cols, x_col="intensity", y="target_inflow_rate"):
    # Same definitions as Section 10: smallest intensity reaching 90% of the seed's
    # maximum gain over no policy, and the grid interval with the steepest slope.
    rows = []
    for keys, g in runs.groupby(group_cols + ["seed"]):
        g = g.sort_values(x_col)
        x, v = g[x_col].to_numpy(), g[y].to_numpy()
        gain = v - v[0]
        k = int(np.argmax(np.diff(v) / np.diff(x)))
        rows.append({**dict(zip(group_cols + ["seed"], keys)),
                     "response90": x[np.argmax(gain >= 0.9 * gain.max())] if gain.max() > 0 else np.nan,
                     "steepest_low": x[k], "steepest_high": x[k + 1]})
    return pd.DataFrame(rows)


def saturation_table(runs, group_cols):
    """Per-group 90% response thresholds plus Hill-fit ceiling of the mean response curve."""
    thr = thresholds_by_seed(runs, group_cols).groupby(group_cols).agg(
        response90_median=("response90", "median"),
        steepest_low_median=("steepest_low", "median"),
    )
    curve = runs.groupby(group_cols + ["intensity"])["target_inflow_rate"].mean().reset_index()
    fit_rows = []
    for keys, g in curve.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        g = g.sort_values("intensity")
        base = float(g["target_inflow_rate"].iloc[0])
        row = dict(zip(group_cols, keys))
        row.update({"baseline": base,
                    "gain_at_1": float(g.loc[np.isclose(g["intensity"], 1.0), "target_inflow_rate"].iloc[0]) - base,
                    "gain_at_max": float(g["target_inflow_rate"].iloc[-1]) - base})
        try:
            y0, A, K, n = fit_hill(g["intensity"], g["target_inflow_rate"])
            row.update({"fitted_ceiling_gain": y0 + A - base, "half_response_K": K, "hill_n": n,
                        "share_of_ceiling_at_max": row["gain_at_max"] / (y0 + A - base) if y0 + A > base else np.nan})
        except RuntimeError:
            row.update({"fitted_ceiling_gain": np.nan, "half_response_K": np.nan, "hill_n": np.nan,
                        "share_of_ceiling_at_max": np.nan})
        fit_rows.append(row)
    return pd.DataFrame(fit_rows).merge(thr.reset_index(), on=group_cols)


def strong_minus_weak(runs, keys, metric="target_inflow_rate"):
    """Paired (strong - weak ties) difference in the combined-policy effect, with 95% t half-width."""
    rows = []
    for k, g in runs.groupby(keys):
        w = g.pivot_table(index=["interaction", "seed"], columns="policy", values=metric)
        eff = w["combined"] - w["baseline"]
        d = (eff.xs("strong") - eff.xs("weak")).to_numpy()
        rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "strong_minus_weak": d.mean(),
                     "ci95": student_t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))})
    return pd.DataFrame(rows)
