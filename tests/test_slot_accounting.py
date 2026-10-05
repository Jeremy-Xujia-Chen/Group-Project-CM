"""Unit tests for housing-slot accounting in PolicySocialABM._migration_step.

Each state starts a year with
    slots = max(1, round(households * vacancy * housing_entry_factor
                         + households * housing_policy * housing_policy_slots))
An accepted household move takes one slot at the destination and frees one at the
origin; a rejected move changes nothing. The slot counts are local to the step, so
each test replays the recorded decisions against an independent slot ledger and
compares the ledger with the ``logistic`` argument the model actually used.
"""
import unittest

import numpy as np

from src.config import STATE_CODES
from src.parameters import policy_table
from tests.helpers import (
    HOT_STATE,
    accept_while_capacity_remains,
    always_accept,
    initial_slots,
    make_model,
    patched_acceptance,
    proposed_rows,
)

IDX = {s: i for i, s in enumerate(STATE_CODES)}


def replay(model, slots, args):
    """Return the logistic arguments the model should have produced, given its decisions."""
    slots = slots.copy()
    capacity = slots.copy()
    p = model.params
    expected = []
    for row in proposed_rows(model):
        dest, origin = IDX[row["chosen_destination"]], IDX[row["origin"]]
        ratio = slots[dest] / max(float(capacity[dest]), 1.0)
        expected.append((ratio - p["housing_capacity_buffer"]) / max(p["housing_market_softness"], 1e-4))
        if row["accepted_move"]:
            slots[dest] -= 1
            slots[origin] += 1
    return np.array(expected), slots


class TestSlotLedger(unittest.TestCase):
    def check(self, model, accept):
        slots = initial_slots(model)
        with patched_acceptance(accept) as args:
            model._migration_step(0)
        expected, final = replay(model, slots, args)
        self.assertGreater(len(args), 5, "too few proposals for a meaningful check")
        np.testing.assert_allclose(args, expected)
        return slots, final, args

    def test_every_accepted_move_takes_and_frees_one_slot(self):
        self.check(make_model(), always_accept)

    def test_rejected_moves_leave_slots_unchanged(self):
        # Scarce housing plus a threshold rule gives a genuine mix of accepts and rejects.
        model = make_model(housing_entry_factor=0.05)
        self.check(model, accept_while_capacity_remains)
        outcomes = {r["accepted_move"] for r in proposed_rows(model)}
        self.assertEqual(outcomes, {0, 1})

    def test_slots_are_conserved_across_moves(self):
        slots, final, _ = self.check(make_model(), always_accept)
        self.assertEqual(final.sum(), slots.sum())


class TestInitialCapacity(unittest.TestCase):
    """Capacity scales with households, vacancy, and housing policy."""

    def test_housing_policy_adds_slots_in_the_target_state_only(self):
        base = make_model(policies=policy_table())
        pol = make_model(policies=policy_table(HOT_STATE, housing=4.0))
        b, p = initial_slots(base), initial_slots(pol)
        self.assertGreater(p[IDX[HOT_STATE]], b[IDX[HOT_STATE]])
        others = [i for s, i in IDX.items() if s != HOT_STATE]
        np.testing.assert_array_equal(p[others], b[others])

    def test_policy_slots_follow_the_formula(self):
        model = make_model(policies=policy_table(HOT_STATE, housing=4.0))
        households = model.agents.groupby("state")["household_id"].nunique()[HOT_STATE]
        vacancy = model.base_state.loc[HOT_STATE, "vacancy_rate"]
        p = model.params
        expected = max(1, round(households * vacancy * p["housing_entry_factor"]
                                + households * 4.0 * p["housing_policy_slots"]))
        self.assertEqual(initial_slots(model)[IDX[HOT_STATE]], expected)

    def test_model_uses_the_policy_capacity(self):
        # The ledger replay only matches if the model's nominal capacity includes the policy term.
        model = make_model(policies=policy_table(HOT_STATE, housing=4.0, migration_incentive=1.5),
                           housing_entry_factor=0.05)
        slots = initial_slots(model)
        with patched_acceptance(always_accept) as args:
            model._migration_step(0)
        expected, _ = replay(model, slots, args)
        self.assertGreater(len(args), 5)
        np.testing.assert_allclose(args, expected)


if __name__ == "__main__":
    unittest.main()
