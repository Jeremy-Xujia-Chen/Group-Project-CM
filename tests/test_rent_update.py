"""Unit tests for the lagged rent update in PolicySocialABM._environment_step.

    rent_growth = clip(0.10 * population_growth
                       - housing_policy_effect * housing_policy
                       + 0.005 * (1 / housing_capacity_index - 1)
                       + rent_rejection_elasticity * latest_rejection_rate,
                       -0.04, 0.12)
    rent_index  = clip(rent_index * (1 + rent_growth), 0.70, 1.80)
"""
import unittest

import numpy as np
import pandas as pd

from src.config import STATE_CODES
from src.parameters import policy_table
from tests.helpers import HOT_STATE, always_reject, make_model, patched_acceptance


def zero_rejections():
    return {s: 0.0 for s in STATE_CODES}


class RentTestCase(unittest.TestCase):
    def setUp(self):
        self.model = make_model(policies=policy_table(), n_agents=200)

    def capacity_term(self):
        return 0.005 * (1.0 / self.model.state["housing_capacity_index"] - 1.0)

    def step_and_ratio(self):
        before = self.model.state["rent_index"].copy()
        self.model._environment_step()
        return self.model.state["rent_index"] / before - 1.0


class TestRentBaseline(RentTestCase):
    def test_no_pressure_leaves_only_the_capacity_drift(self):
        growth = self.step_and_ratio()
        np.testing.assert_allclose(growth, self.capacity_term())

    def test_empty_rejection_history_is_treated_as_zero_pressure(self):
        self.model.housing_rejection_by_state = {}
        np.testing.assert_allclose(self.step_and_ratio(), self.capacity_term())

    def test_scarce_housing_capacity_index_raises_rent_cheap_housing_lowers_it(self):
        growth = self.step_and_ratio()
        hci = self.model.state["housing_capacity_index"]
        self.assertTrue((growth[hci < 1] > 0).all())
        self.assertTrue((growth[hci > 1] < 0).all())

    def test_base_state_is_not_modified(self):
        before = self.model.base_state["rent_index"].copy()
        self.model._environment_step()
        pd.testing.assert_series_equal(self.model.base_state["rent_index"], before)


class TestRentRejectionPressure(RentTestCase):
    def test_rejection_rate_raises_rent_by_elasticity_times_rate(self):
        self.model.housing_rejection_by_state = {0: {**zero_rejections(), HOT_STATE: 0.5}}
        growth = self.step_and_ratio()
        expected = self.capacity_term()
        expected[HOT_STATE] += self.model.params["rent_rejection_elasticity"] * 0.5
        np.testing.assert_allclose(growth, expected)

    def test_only_rejecting_state_is_affected(self):
        self.model.housing_rejection_by_state = {0: {**zero_rejections(), HOT_STATE: 1.0}}
        growth = self.step_and_ratio()
        others = [s for s in STATE_CODES if s != HOT_STATE]
        np.testing.assert_allclose(growth[others], self.capacity_term()[others])

    def test_zero_elasticity_switches_the_channel_off(self):
        model = make_model(policies=policy_table(), n_agents=200, rent_rejection_elasticity=0.0)
        model.housing_rejection_by_state = {0: {s: 1.0 for s in STATE_CODES}}
        before = model.state["rent_index"].copy()
        capacity = 0.005 * (1.0 / model.state["housing_capacity_index"] - 1.0)
        model._environment_step()
        np.testing.assert_allclose(model.state["rent_index"] / before - 1.0, capacity)

    def test_only_the_latest_year_of_rejections_is_used(self):
        self.model.housing_rejection_by_state = {
            0: {s: 1.0 for s in STATE_CODES},
            1: zero_rejections(),
        }
        np.testing.assert_allclose(self.step_and_ratio(), self.capacity_term())

    def test_nan_rejection_rate_is_treated_as_zero(self):
        self.model.housing_rejection_by_state = {0: {**zero_rejections(), HOT_STATE: np.nan}}
        np.testing.assert_allclose(self.step_and_ratio(), self.capacity_term())


class TestRentPolicyAndPopulation(RentTestCase):
    def test_housing_policy_lowers_rent_growth_in_target_state_only(self):
        model = make_model(policies=policy_table(HOT_STATE, housing=1.0), n_agents=200)
        before = model.state["rent_index"].copy()
        capacity = 0.005 * (1.0 / model.state["housing_capacity_index"] - 1.0)
        model._environment_step()
        growth = model.state["rent_index"] / before - 1.0
        expected = capacity.copy()
        expected[HOT_STATE] -= model.params["housing_policy_effect"] * 1.0
        np.testing.assert_allclose(growth, expected)

    def test_population_growth_raises_rent(self):
        # Pretend the state had 10% fewer residents last year.
        counts = self.model._counts()
        self.model.previous_counts = counts.copy()
        self.model.previous_counts[HOT_STATE] = counts[HOT_STATE] / 1.1
        growth = self.step_and_ratio()
        expected = self.capacity_term()
        expected[HOT_STATE] += 0.10 * 0.1
        np.testing.assert_allclose(growth, expected)

    def test_previous_counts_are_advanced_to_current(self):
        self.model.previous_counts = self.model._counts() * 0 + 1.0
        self.model._environment_step()
        pd.testing.assert_series_equal(self.model.previous_counts, self.model._counts())


class TestRentClipping(RentTestCase):
    def test_annual_growth_is_capped_at_12_percent(self):
        model = make_model(policies=policy_table(), n_agents=200, rent_rejection_elasticity=5.0)
        model.housing_rejection_by_state = {0: {s: 1.0 for s in STATE_CODES}}
        before = model.state["rent_index"].copy()
        model._environment_step()
        np.testing.assert_allclose(model.state["rent_index"] / before - 1.0, 0.12)

    def test_annual_fall_is_capped_at_4_percent(self):
        model = make_model(policies=policy_table(HOT_STATE, housing=50.0), n_agents=200)
        before = model.state["rent_index"][HOT_STATE]
        model._environment_step()
        self.assertAlmostEqual(model.state["rent_index"][HOT_STATE] / before - 1.0, -0.04)

    def test_rent_index_cannot_exceed_upper_bound(self):
        self.model.state["rent_index"] = 1.79
        self.model.housing_rejection_by_state = {0: {s: 1.0 for s in STATE_CODES}}
        self.model._environment_step()
        self.assertTrue((self.model.state["rent_index"] <= 1.80).all())
        self.assertAlmostEqual(self.model.state["rent_index"].max(), 1.80)

    def test_rent_index_cannot_fall_below_lower_bound(self):
        model = make_model(policies=policy_table(HOT_STATE, housing=50.0), n_agents=200)
        model.state["rent_index"] = 0.71
        model._environment_step()
        self.assertAlmostEqual(model.state["rent_index"][HOT_STATE], 0.70)


class TestRentFeedbackFromMigrationStep(RentTestCase):
    """Rejections produced by the migration step feed into the next rent update."""

    def test_rejected_inflow_raises_next_year_rent_in_destination(self):
        model = make_model()
        with patched_acceptance(always_reject):
            model._migration_step(0)
        rates = model.housing_rejection_by_state[0]
        rejecting = [s for s in STATE_CODES if rates[s] > 0]
        self.assertTrue(rejecting)
        before = model.state["rent_index"].copy()
        capacity = 0.005 * (1.0 / model.state["housing_capacity_index"] - 1.0)
        model._environment_step()
        growth = model.state["rent_index"] / before - 1.0
        for s in STATE_CODES:
            self.assertGreaterEqual(growth[s], capacity[s] - 1e-12)
        for s in rejecting:
            self.assertAlmostEqual(growth[s], min(capacity[s] + 0.15 * rates[s], 0.12))


if __name__ == "__main__":
    unittest.main()
