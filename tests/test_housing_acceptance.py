"""Unit tests for housing acceptance of proposed moves (PolicySocialABM._migration_step)."""
import unittest
from unittest import mock

import numpy as np

from src import parameters
from src.config import STATE_CODES
from tests.helpers import (
    HOT_STATE,
    accept_while_capacity_remains,
    always_accept,
    always_reject,
    household_sizes,
    make_model,
    patched_acceptance,
    proposed_rows,
)


class TestForcedRejection(unittest.TestCase):
    """Boundary case: no housing is ever available (acceptance probability 0)."""

    def setUp(self):
        self.model = make_model()
        self.before = self.model.agents["state"].copy()
        with patched_acceptance(always_reject):
            self.old, self.new, self.proposed, self.rejected = self.model._migration_step(0)

    def test_some_moves_were_proposed(self):
        # Guards the other assertions against passing vacuously.
        self.assertGreater(self.proposed, 5)

    def test_every_proposed_move_is_rejected(self):
        self.assertEqual(self.rejected, self.proposed)

    def test_nobody_changes_state(self):
        self.assertTrue((self.model.agents["state"] == self.before).all())
        self.assertTrue((self.old == self.new).all())

    def test_no_move_events_are_recorded(self):
        self.assertEqual(self.model.move_history, [])

    def test_decision_rows_flag_rejections(self):
        rows = proposed_rows(self.model)
        self.assertEqual(len(rows), self.proposed)
        self.assertTrue(all(r["housing_rejected"] == 1 and r["accepted_move"] == 0 for r in rows))

    def test_rejection_rate_is_one_where_proposals_exist_else_zero(self):
        rates = self.model.housing_rejection_by_state[0]
        proposals = {s: 0 for s in STATE_CODES}
        for r in proposed_rows(self.model):
            proposals[r["chosen_destination"]] += 1
        for state in STATE_CODES:
            expected = 1.0 if proposals[state] else 0.0  # 0 proposals must not divide by zero
            self.assertEqual(rates[state], expected, state)


class TestForcedAcceptance(unittest.TestCase):
    """Boundary case: housing is always available (acceptance probability 1)."""

    def setUp(self):
        self.model = make_model()
        self.sizes = household_sizes(self.model)
        with patched_acceptance(always_accept):
            self.old, self.new, self.proposed, self.rejected = self.model._migration_step(0)

    def test_nothing_is_rejected(self):
        self.assertGreater(self.proposed, 5)
        self.assertEqual(self.rejected, 0)
        self.assertTrue(all(r["housing_rejected"] == 0 for r in self.model.decision_rows))

    def test_accepted_households_arrive_at_chosen_destination(self):
        agents = self.model.agents
        for row in proposed_rows(self.model):
            members = agents[agents["household_id"] == row["household_id"]]
            self.assertTrue((members["state"] == row["chosen_destination"]).all())

    def test_household_members_move_together(self):
        agents = self.model.agents
        self.assertTrue((agents.groupby("household_id")["state"].nunique() == 1).all())

    def test_move_events_are_person_level(self):
        # One move event per person, so a 2-adult household contributes 2 events.
        expected = sum(self.sizes[r["household_id"]] for r in proposed_rows(self.model))
        events = sum(len(df) for df in self.model.move_history)
        self.assertEqual(events, expected)

    def test_stayers_are_not_proposals(self):
        stayers = [r for r in self.model.decision_rows if r["proposed_move"] == 0]
        self.assertTrue(stayers)
        self.assertTrue(all(r["origin"] == r["chosen_destination"] for r in stayers))
        self.assertTrue(all(r["accepted_move"] == 0 and r["housing_rejected"] == 0 for r in stayers))


class TestAcceptanceProbabilityInput(unittest.TestCase):
    """The logistic argument must be (remaining/capacity - buffer) / softness."""

    def test_first_inflow_into_a_state_sees_full_capacity(self):
        model = make_model()
        with patched_acceptance(always_accept) as args:
            model._migration_step(0)
        p = model.params
        full = (1.0 - p["housing_capacity_buffer"]) / p["housing_market_softness"]
        first, freed = {}, set()
        for row, x in zip(proposed_rows(model), args):
            dest = row["chosen_destination"]
            # A state that already lost a household has spare slots, so skip it.
            if dest not in first and dest not in freed:
                first[dest] = x
            freed.add(row["origin"])
        self.assertTrue(first)
        for state, x in first.items():
            self.assertAlmostEqual(x, full, msg=state)

    def test_logistic_called_once_per_proposal_and_never_for_stayers(self):
        model = make_model()
        with patched_acceptance(always_accept) as args:
            _, _, proposed, _ = model._migration_step(0)
        self.assertEqual(len(args), proposed)

    def test_softness_and_buffer_are_read_from_params(self):
        model = make_model(housing_market_softness=0.5, housing_capacity_buffer=0.2)
        with patched_acceptance(always_accept) as args:
            model._migration_step(0)
        self.assertAlmostEqual(args[0], (1.0 - 0.2) / 0.5)

    def test_zero_softness_is_floored_to_avoid_division_by_zero(self):
        model = make_model(housing_market_softness=0.0)
        with patched_acceptance(always_accept) as args:
            model._migration_step(0)
        self.assertTrue(np.isfinite(args).all())
        self.assertAlmostEqual(args[0], (1.0 - 0.05) / 1e-4)


class TestRealAcceptanceCurve(unittest.TestCase):
    """Unpatched behaviour: plentiful housing accepts, scarce housing rejects."""

    def rejection_rate(self, **overrides):
        proposed = rejected = 0
        for seed in range(1, 6):
            model = make_model(seed=seed, **overrides)
            _, _, p, r = model._migration_step(0)
            proposed += p
            rejected += r
        self.assertGreater(proposed, 20)
        return rejected / proposed

    def test_plentiful_housing_almost_never_rejects(self):
        self.assertLess(self.rejection_rate(housing_entry_factor=50.0), 0.02)

    def test_scarce_housing_rejects_more_than_plentiful(self):
        scarce = self.rejection_rate(housing_entry_factor=0.05)
        plentiful = self.rejection_rate(housing_entry_factor=50.0)
        self.assertGreater(scarce, plentiful + 0.3)

    def test_rejection_rates_are_proportions(self):
        model = make_model()
        model._migration_step(0)
        for v in model.housing_rejection_by_state[0].values():
            self.assertTrue(0.0 <= v <= 1.0)


class TestFocalStateMechanismCounts(unittest.TestCase):
    """Household-level proposal/rejection counts vs person-level accepted counts."""

    def run_step(self, accept):
        model = make_model()
        sizes = household_sizes(model)
        with mock.patch.object(parameters, "FOCAL_STATE", HOT_STATE), patched_acceptance(accept):
            model._migration_step(0)
        return model, sizes, model.mechanism_rows[0]

    def test_counts_match_decision_rows(self):
        model, _, mech = self.run_step(accept_while_capacity_remains)
        rows = [r for r in proposed_rows(model) if r["chosen_destination"] == HOT_STATE]
        self.assertGreater(len(rows), 0)
        self.assertEqual(mech["target_proposed"], len(rows))
        self.assertEqual(mech["target_housing_rejected"], sum(r["housing_rejected"] for r in rows))

    def test_accepted_is_counted_per_person_not_per_household(self):
        model, sizes, mech = self.run_step(always_accept)
        rows = [r for r in proposed_rows(model) if r["chosen_destination"] == HOT_STATE]
        self.assertEqual(mech["target_accepted"], sum(sizes[r["household_id"]] for r in rows))
        self.assertEqual(mech["target_social_cost_n"], mech["target_accepted"])

    def test_rejected_moves_contribute_no_accepted_agents(self):
        _, _, mech = self.run_step(always_reject)
        self.assertEqual(mech["target_accepted"], 0)
        self.assertEqual(mech["target_housing_rejected"], mech["target_proposed"])


if __name__ == "__main__":
    unittest.main()
