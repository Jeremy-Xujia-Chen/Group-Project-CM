"""Shared fixtures for the housing tests.

The housing-acceptance step in ``PolicySocialABM._migration_step`` keeps its slot
counts in local variables, so the tests observe it through ``src.model.logistic``:
that function is called exactly once per proposed move, with the argument
``(remaining_slots / nominal_capacity - buffer) / softness``. ``patched_acceptance``
swaps it for a recorder whose return value (the acceptance probability) the test
controls, which makes accept/reject deterministic.
"""
from contextlib import contextmanager
from unittest import mock

import numpy as np

from src.config import STATE_CODES
from src.data import state_data
from src.model import PolicySocialABM
from src.parameters import PARAMS, policy_table

# Enough agents and pull towards one state that several moves are proposed per year.
N_AGENTS = 600
HOT_STATE = "NY"


def make_model(policies=None, seed=1, n_agents=N_AGENTS, **param_overrides):
    params = PARAMS.copy()
    params.update(param_overrides)
    if policies is None:
        policies = policy_table(state_incentives={HOT_STATE: 1.5})
    return PolicySocialABM(state_data, policies, n_agents=n_agents, years=1, seed=seed, params=params)


def initial_slots(model):
    """Independent re-statement of the slot formula, evaluated before the migration step."""
    households = model.agents.groupby("state")["household_id"].nunique()
    households = households.reindex(STATE_CODES, fill_value=0).to_numpy()
    vacancy = model.base_state["vacancy_rate"].to_numpy()
    policy = model.policies["housing_policy"].to_numpy()
    raw = (
        households * vacancy * model.params["housing_entry_factor"]
        + households * policy * model.params["housing_policy_slots"]
    )
    return np.maximum(1, np.round(raw)).astype(int)


def household_sizes(model):
    return model.agents.groupby("household_id").size()


@contextmanager
def patched_acceptance(accept):
    """Record every argument passed to ``logistic`` and return ``accept(x)`` as P(accept).

    Yields the list of recorded arguments (one per proposed move, in processing order).
    """
    seen = []

    def fake(x):
        seen.append(float(x))
        return float(accept(float(x)))

    with mock.patch("src.model.logistic", side_effect=fake):
        yield seen


def always_accept(_x):
    return 1.0


def always_reject(_x):
    return 0.0


def accept_while_capacity_remains(x):
    """Accept iff remaining/nominal >= buffer, i.e. the logistic argument is non-negative."""
    return 1.0 if x >= 0 else 0.0


def proposed_rows(model):
    return [r for r in model.decision_rows if r["proposed_move"] == 1]
