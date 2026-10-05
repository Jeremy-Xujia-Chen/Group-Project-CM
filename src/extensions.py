"""Model variants used in the robustness checks (Sections 13.4 and 13.5)."""
import numpy as np
import pandas as pd

from src import parameters
from src.calibration import DESTINATION_EFFECTS
from src.config import BASE_YEAR, EXTERNAL_ORIGIN_TARGET, STATE_CODES
from src.data import distance_index
from src.model import PolicySocialABM
from utils.stats import logistic


class DistanceDecayABM(PolicySocialABM):
    def __init__(self, *args, network_decay=3.0, **kwargs):
        self.network_decay = float(network_decay)
        super().__init__(*args, **kwargs)
        rng = np.random.default_rng(self.seed + 7000)
        for i in range(len(self.agents)):
            self.network[i] = self._decay_contacts(i, rng)

    def _decay_weights(self, i, candidates):
        d = distance_index.to_numpy()
        s = self._state_indices()
        w = np.exp(-self.network_decay * d[s[i], s[candidates]] / d.max())
        return w / w.sum()

    def _decay_contacts(self, i, rng):
        pool = np.delete(np.arange(len(self.agents)), i)
        return rng.choice(pool, size=self.network_k, replace=False, p=self._decay_weights(i, pool))

    def _rewire_agent(self, i, t):
        rng = self._rng(71 + i % 11, t + i)
        n = len(self.agents)
        self.network[i] = self._decay_contacts(i, rng)
        rows, cols = np.where(self.network == i)
        for r, c in zip(rows, cols):
            candidates = np.setdiff1d(np.arange(n), [r, i])
            self.network[r, c] = int(rng.choice(candidates, p=self._decay_weights(r, candidates)))


class OpenSystemABM(PolicySocialABM):
    def __init__(self, *args, external_rate=0.05, capacity_mode="shared", **kwargs):
        self.external_rate = float(external_rate)
        self.capacity_mode = capacity_mode
        self.external_rows = []
        super().__init__(*args, **kwargs)

    def _housing_slots(self):
        hh = self.agents.groupby("state")["household_id"].nunique().reindex(STATE_CODES, fill_value=0).to_numpy()
        return np.maximum(1, np.round(
            hh * self.base_state["vacancy_rate"].to_numpy() * self.params["housing_entry_factor"]
            + hh * self.policies["housing_policy"].to_numpy() * self.params["housing_policy_slots"])).astype(int)

    def _outsider_utility(self):
        wage, jobs, rent = self.state["wage_index"].to_numpy(), self.state["job_index"].to_numpy(), self.state["rent_index"].to_numpy()
        income_anchor = 0.65 * self.base_state["median_household_income"].to_numpy() * (wage / self.base_state["wage_index"].to_numpy())
        return (self.params["beta_income"] * np.log(income_anchor / np.median(income_anchor))
                + self.params["beta_jobs"] * jobs - self.params["beta_rent"] * rent
                + DESTINATION_EFFECTS.to_numpy()
                + self.params["beta_housing_signal"] * self.policies["housing_policy"].to_numpy()
                + self.params["beta_employment_signal"] * self.policies["employment_incentive"].to_numpy()
                + self.params["beta_migration_incentive"] * self.policies["migration_incentive"].to_numpy())

    def _external_exchange(self, t, remaining, nominal, internal_prop, internal_rej):
        a = self.agents; S = len(STATE_CODES); rng = self._rng(97, t)
        z = self._outsider_utility() / self.params["temperature"]
        p = np.exp(z - z.max()); p /= p.sum()
        n_try = int(rng.poisson(self.external_rate * len(a)))
        softness = max(float(self.params.get("housing_market_softness", 0.10)), 1e-4)
        buffer = float(self.params.get("housing_capacity_buffer", 0.05))
        prop = np.zeros(S, dtype=int); rej = np.zeros(S, dtype=int); acc = np.zeros(S, dtype=int)
        arrivals = []
        for dest in rng.choice(S, size=n_try, p=p):
            prop[dest] += 1
            if rng.random() > float(logistic((remaining[dest] / max(nominal[dest], 1.0) - buffer) / softness)):
                rej[dest] += 1
            else:
                acc[dest] += 1; remaining[dest] -= 1; arrivals.append(int(dest))
        leavers = rng.choice(len(a), size=len(arrivals), replace=False) if arrivals else np.array([], dtype=int)
        pop_p = self.base_state["young_adult_population_est_18_35"].to_numpy(dtype=float); pop_p /= pop_p.sum()
        for i, dest in zip(leavers, arrivals):
            state = STATE_CODES[dest]
            age = int(np.clip(np.rint(rng.normal(26, 4.5)), 18, 35))
            college_p = self.base_state.loc[state, "college_share_proxy"] * (0.2 if age < 22 else 0.6 if age < 25 else 1.0)
            college = int(rng.binomial(1, np.clip(college_p, 0.03, 0.75)))
            skill = float(np.clip(rng.normal(1.0 + 0.18 * college, 0.15), 0.55, 1.65))
            a.loc[i, ["age", "college", "skill", "state", "home_state", "is_international", "employed", "has_children",
                      "dependent_children", "unemployed_years", "job_rejected_last_year", "last_move_year"]] = [
                age, college, skill, state, rng.choice(STATE_CODES, p=pop_p), 0, 0, 0, 0, 0, 0, t]
            a.loc[i, "income"] = (0.65 * self.base_state.loc[state, "median_household_income"]
                                  * (0.72 + 0.28 * skill) * (1 + 0.22 * college) * rng.lognormal(0, 0.12))
            a.loc[i, "household_id"] = self.next_household_id; self.next_household_id += 1
            self._rewire_agent(int(i), t)
        # Price feedback sees insider and outsider rejections together.
        tot_p, tot_r = internal_prop + prop, internal_rej + rej
        self.housing_rejection_by_state[t] = {s: tot_r[j] / max(tot_p[j], 1) for j, s in enumerate(STATE_CODES)}
        f = STATE_CODES.index(parameters.FOCAL_STATE)
        self.external_rows.append({"year": BASE_YEAR + t + 1, "external_attempts": n_try, "external_accepted": int(acc.sum()),
                                   "target_external_proposed": int(prop[f]), "target_external_accepted": int(acc[f]),
                                   "target_external_rejected": int(rej[f])})

    def step(self, t):
        self._demography_step(t)
        self._household_formation_step(t)
        births = self._family_step(t)
        slots0 = self._housing_slots()
        hh_state = self.agents.groupby("household_id")["state"].first()
        n_dec = len(self.decision_rows)
        old_states, new_states, proposed, rejected = self._migration_step(t)
        new_hh_state = self.agents.groupby("household_id")["state"].first().reindex(hh_state.index)
        moved = hh_state != new_hh_state
        into = new_hh_state[moved].value_counts().reindex(STATE_CODES, fill_value=0).to_numpy()
        outof = hh_state[moved].value_counts().reindex(STATE_CODES, fill_value=0).to_numpy()
        factor = 1.0 / (1.0 - EXTERNAL_ORIGIN_TARGET) if self.capacity_mode == "scaled" else 1.0
        remaining = (slots0 - into + outof + slots0 * (factor - 1.0)).astype(float)
        dec = pd.DataFrame(self.decision_rows[n_dec:])
        prop_int = dec[dec["proposed_move"] == 1].groupby("chosen_destination").size().reindex(STATE_CODES, fill_value=0).to_numpy()
        rej_int = dec[dec["housing_rejected"] == 1].groupby("chosen_destination").size().reindex(STATE_CODES, fill_value=0).to_numpy()
        self._external_exchange(t, remaining, slots0 * factor, prop_int, rej_int)
        self._employment_step(t)
        self._environment_step()
        self._record_metrics(t, old_states, new_states, proposed, rejected, births)
        if t + 1 in {5, 10, 15, 20}:
            self._snapshot(BASE_YEAR + t + 1)
