"""IRS route calibration of destination-choice coefficients (Section 2).

Fits the utility coefficients and destination fixed effects to observed IRS
route shares, validates on held-out origins, then refits on all origins.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import STATE_CODES
from src.data import distance_index, irs_flows, state_data

STATE_POS = {s: i for i, s in enumerate(STATE_CODES)}

PRIOR_BETA = np.array([
    1.10,  # income
    1.25,  # jobs
    1.00,  # rent
    0.20,  # distance
    0.45,  # education
    0.50,  # childcare
], dtype=float)

PRIOR_SCALE = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
CALIBRATION_TEMPERATURE = 0.85

def calibration_features():
    return {
        "log_income": np.log(
            state_data["wage_index"].to_numpy()
        ),
        "jobs": state_data["job_index"].to_numpy(),
        "rent": state_data["rent_index"].to_numpy(),
        "education": state_data[
            "education_access_index"
        ].to_numpy(),
        "childcare": (
            1.0
            / state_data["childcare_cost_index"]
        ).to_numpy(),
    }

CAL_FEATURES = calibration_features()

def predicted_route_shares(theta, origins):
    """
    Deterministic expected route shares.

    Rather than representing each origin with one average person,
    the calibration mixes four composition groups:
      college / non-college × child / no-child.

    This better matches the heterogeneous utility function used in
    the ABM while keeping calibration deterministic.
    """
    beta = theta[:6]
    destination_effect = np.r_[0.0, theta[6:]]

    rows = []

    for origin in origins:
        observed = irs_flows[
            irs_flows["origin"] == origin
        ].copy()
        destinations = observed[
            "destination"
        ].tolist()

        college_share = float(
            state_data.loc[
                origin, "college_share_proxy"
            ]
        ) * 0.82
        college_share = float(
            np.clip(college_share, 0.05, 0.70)
        )

        child_share = 0.25

        mixture_probability = np.zeros(
            len(destinations), dtype=float
        )

        for college, college_weight in [
            (0, 1.0 - college_share),
            (1, college_share),
        ]:
            for has_child, child_weight in [
                (0, 1.0 - child_share),
                (1, child_share),
            ]:
                utility = []

                for destination in destinations:
                    d = STATE_POS[destination]

                    u = (
                        beta[0]
                        * CAL_FEATURES[
                            "log_income"
                        ][d]
                        + beta[1]
                        * CAL_FEATURES[
                            "jobs"
                        ][d]
                        - beta[2]
                        * CAL_FEATURES[
                            "rent"
                        ][d]
                        - beta[3]
                        * distance_index.loc[
                            origin, destination
                        ]
                        + beta[4]
                        * (1 - college)
                        * CAL_FEATURES[
                            "education"
                        ][d]
                        + beta[5]
                        * has_child
                        * CAL_FEATURES[
                            "childcare"
                        ][d]
                        + destination_effect[d]
                    )
                    utility.append(u)

                utility = (
                    np.asarray(utility)
                    / CALIBRATION_TEMPERATURE
                )
                p = np.exp(
                    utility - utility.max()
                )
                p /= p.sum()

                mixture_probability += (
                    college_weight
                    * child_weight
                    * p
                )

        mixture_probability /= (
            mixture_probability.sum()
        )

        for (_, row), pred in zip(
            observed.iterrows(),
            mixture_probability,
        ):
            rows.append({
                "origin": origin,
                "destination": row[
                    "destination"
                ],
                "returns": row["returns"],
                "observed_share": row[
                    "observed_share"
                ],
                "predicted_share": pred,
            })

    return pd.DataFrame(rows)

def calibration_loss(
    theta,
    origins,
    lambda_beta=0.02,
    lambda_fe=0.05,
):
    comp = predicted_route_shares(
        theta, origins
    )

    weights = np.sqrt(
        comp["returns"].to_numpy()
    )
    sq_error = (
        comp["predicted_share"]
        - comp["observed_share"]
    ) ** 2

    fit_loss = np.average(
        sq_error, weights=weights
    )

    beta_penalty = lambda_beta * np.mean(
        (
            (theta[:6] - PRIOR_BETA)
            / PRIOR_SCALE
        ) ** 2
    )
    fe_penalty = (
        lambda_fe
        * np.mean(theta[6:] ** 2)
    )

    return float(
        fit_loss
        + beta_penalty
        + fe_penalty
    )

def route_fit_metrics(theta, origins):
    comp = predicted_route_shares(
        theta, origins
    )
    w = np.sqrt(
        comp["returns"].to_numpy()
    )

    error = (
        comp["predicted_share"]
        - comp["observed_share"]
    )

    rmse = np.sqrt(
        np.average(
            error ** 2,
            weights=w,
        )
    )
    mae = np.average(
        np.abs(error),
        weights=w,
    )

    if (
        comp[
            "predicted_share"
        ].std()
        > 0
    ):
        corr = np.corrcoef(
            comp["observed_share"],
            comp["predicted_share"],
        )[0, 1]
    else:
        corr = np.nan

    return {
        "weighted_rmse": float(rmse),
        "weighted_mae": float(mae),
        "correlation": float(corr),
    }

x0 = np.r_[
    PRIOR_BETA,
    np.zeros(7),
]

bounds = (
    [
        (0.0, 4.0),
        (0.0, 4.0),
        (0.0, 4.0),
        (0.0, 3.0),
        (0.0, 2.0),
        (0.0, 2.0),
    ]
    + [(-2.0, 2.0)] * 7
)

TRAIN_ORIGINS = [
    "CA", "TX", "NY",
    "FL", "MA", "NC",
]
VALIDATION_ORIGINS = [
    "IL", "WA",
]

train_result = minimize(
    calibration_loss,
    x0,
    args=(TRAIN_ORIGINS,),
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": 500},
)

validation_table = pd.DataFrame([
    {
        "model": "Prior specification",
        "sample": "train",
        **route_fit_metrics(
            x0, TRAIN_ORIGINS
        ),
    },
    {
        "model": "Calibrated on train",
        "sample": "train",
        **route_fit_metrics(
            train_result.x,
            TRAIN_ORIGINS,
        ),
    },
    {
        "model": "Prior specification",
        "sample": "holdout",
        **route_fit_metrics(
            x0,
            VALIDATION_ORIGINS,
        ),
    },
    {
        "model": "Calibrated on train",
        "sample": "holdout",
        **route_fit_metrics(
            train_result.x,
            VALIDATION_ORIGINS,
        ),
    },
])


# Final fit on all origins; these coefficients feed PARAMS.
final_calibration = minimize(
    calibration_loss,
    x0,
    args=(STATE_CODES,),
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": 700},
)

CALIBRATED_BETA = final_calibration.x[:6].copy()
DESTINATION_EFFECTS = pd.Series(
    np.r_[0.0, final_calibration.x[6:]],
    index=STATE_CODES,
    name="destination_effect",
)

final_route_metrics = route_fit_metrics(
    final_calibration.x, STATE_CODES
)
