"""Core agent-based model: household migration with social networks, capacity limits and feedback (Section 6)."""
import numpy as np
import pandas as pd

from src import parameters
from src.agents import initialise_agents
from src.calibration import DESTINATION_EFFECTS
from src.config import (
    BASE_YEAR,
    EXPERIMENT_AGENTS,
    EXPERIMENT_YEARS,
    RANDOM_SEED,
    SOCIAL_UTILITY_WEIGHTS,
    STATE_CODES,
)
from src.data import distance_index
from src.parameters import PARAMS, mortality_probability
from utils.stats import logistic


class PolicySocialABM:
    def __init__(
        self,
        state_df,
        policies,
        n_agents=EXPERIMENT_AGENTS,
        years=EXPERIMENT_YEARS,
        seed=RANDOM_SEED,
        params=None,
        interaction_strength=1.0,
        shocks=None,
        entrant_home_local=False,
        entrant_network_home_bias=False,
        network_homophily_share=0.0,
        record_events=True,
    ):
        self.base_state = state_df.copy()
        self.state = state_df.copy()
        self.policies = policies.copy()
        self.n_agents = int(n_agents)
        self.years = int(years)
        self.seed = int(seed)
        self.params = PARAMS.copy() if params is None else params.copy()
        self.interaction_strength = float(interaction_strength)
        self.shocks = {} if shocks is None else dict(shocks)
        self.entrant_home_local = bool(entrant_home_local)
        self.entrant_network_home_bias = bool(entrant_network_home_bias)
        self.network_homophily_share = float(np.clip(network_homophily_share, 0.0, 1.0))
        self.record_events = bool(record_events)

        self.params["beta_peer_network"] = PARAMS["beta_peer_network"] * self.interaction_strength
        self.params["beta_coorigin"] = PARAMS["beta_coorigin"] * self.interaction_strength
        self.params["beta_social_dissatisfaction"] = PARAMS["beta_social_dissatisfaction"] * self.interaction_strength

        self.agents, self.network = initialise_agents(
            state_df,
            n_agents=self.n_agents,
            seed=self.seed + 1000,
            network_k=self.params["network_k"],
            network_homophily_share=self.network_homophily_share,
        )
        self.network_k = self.params["network_k"]
        self.next_household_id = int(self.agents["household_id"].max() + 1)

        self.history = []
        self.move_history = []
        self.decision_rows = []
        self.mechanism_rows = []
        self.demography_rows = []
        self.snapshots = {}
        self.job_rejection_by_state = {}
        self.unfilled_jobs_by_state = {}
        self.housing_rejection_by_state = {}
        self.previous_counts = self._counts()
        self.initial_counts = self.previous_counts.copy()

        self._employment_step(-1)
        self._snapshot(BASE_YEAR)

    def _rng(self, code, t):
        return np.random.default_rng(np.random.SeedSequence([self.seed, int(code), int(t) + 100]))

    def _counts(self):
        return self.agents["state"].value_counts().reindex(STATE_CODES, fill_value=0).astype(float)

    def _state_indices(self):
        return pd.Categorical(self.agents["state"], categories=STATE_CODES).codes

    def _home_indices(self):
        return pd.Categorical(self.agents["home_state"], categories=STATE_CODES).codes

    def _shock_factors(self, t):
        if not self.shocks:
            return {
                "job_slot_multiplier": 1.0,
                "wage_multiplier": 1.0,
                "mortality_multiplier": 1.0,
                "migration_cost_multiplier": 1.0,
            }

        start = int(self.shocks.get("start", 0))
        duration = int(self.shocks.get("duration", 0))
        recovery = int(self.shocks.get("recovery_years", 0))

        if t < start:
            intensity = 0.0
        elif t < start + duration:
            intensity = 1.0
        elif recovery > 0 and t < start + duration + recovery:
            elapsed = t - (start + duration) + 1
            intensity = max(0.0, 1.0 - elapsed / (recovery + 1))
        else:
            intensity = 0.0

        def blend(key):
            target = float(self.shocks.get(key, 1.0))
            return 1.0 + intensity * (target - 1.0)

        return {
            "job_slot_multiplier": blend("job_slot_multiplier"),
            "wage_multiplier": blend("wage_multiplier"),
            "mortality_multiplier": blend("mortality_multiplier"),
            "migration_cost_multiplier": blend("migration_cost_multiplier"),
        }

    def _household_formation_step(self, t):
        a = self.agents
        rng = self._rng(15, t)
        hh = a["household_id"].to_numpy()
        sizes = a["household_id"].map(a["household_id"].value_counts()).to_numpy()
        eligible = (a["age"].to_numpy() >= 22) & (sizes == 1)
        states = a["state"].to_numpy()
        for state in STATE_CODES:
            candidates = np.where(eligible & (states == state))[0]
            if len(candidates) < 2:
                continue
            rng.shuffle(candidates)
            n_adults = int(len(candidates) * self.params["annual_pair_formation_rate"])
            n_adults -= n_adults % 2
            for i, j in candidates[:n_adults].reshape(-1, 2):
                hid = self.next_household_id
                self.next_household_id += 1
                a.loc[[i, j], "household_id"] = hid
                dep = max(int(a.loc[i, "dependent_children"]), int(a.loc[j, "dependent_children"]))
                if dep > 0:
                    a.loc[[i, j], "has_children"] = 1
                    a.loc[[i, j], "dependent_children"] = dep

    def _family_step(self, t):
        a = self.agents
        rng = self._rng(17, t)
        grouped = a.groupby("household_id", sort=False)
        hh_age = grouped["age"].mean()
        hh_children = grouped["dependent_children"].max()
        eligible = hh_age.index[
            (hh_age.to_numpy() >= 20)
            & (hh_age.to_numpy() <= 35)
            & (hh_children.to_numpy() < self.params["max_dependent_children"])
        ]
        if len(eligible) == 0:
            return 0
        ages = hh_age.loc[eligible].to_numpy()
        existing = hh_children.loc[eligible].to_numpy()
        age_profile = np.exp(-0.5 * ((ages - self.params["birth_age_center"]) / self.params["birth_age_spread"]) ** 2)
        p = np.clip(self.params["birth_peak_probability"] * age_profile * (1.0 - 0.18 * existing), 0, 0.09)
        selected = eligible[rng.random(len(eligible)) < p]
        for hid in selected:
            mask = a["household_id"] == hid
            current = int(a.loc[mask, "dependent_children"].max())
            a.loc[mask, "has_children"] = 1
            a.loc[mask, "dependent_children"] = min(current + 1, self.params["max_dependent_children"])
        return int(len(selected))


    def _rewire_agent(self, i, t):
        rng = self._rng(71 + i % 11, t + i)
        n = len(self.agents)
        all_idx = np.arange(n)
        pool = all_idx[all_idx != i]
        home = self.agents["home_state"].to_numpy()

        # Preserve the model-level homophily treatment for replacement entrants.
        # The legacy entrant_network_home_bias switch implies at least 50% same-home
        # contacts but does not overwrite a stronger explicit homophily setting.
        target_homophily = self.network_homophily_share
        if self.entrant_network_home_bias:
            target_homophily = max(target_homophily, 0.50)

        contacts = []
        k_home = int(round(self.network_k * target_homophily))
        if k_home > 0:
            same_home = pool[home[pool] == home[i]]
            if len(same_home):
                take = min(k_home, len(same_home))
                contacts.extend(rng.choice(same_home, size=take, replace=False).tolist())

        remaining = self.network_k - len(contacts)
        if remaining > 0:
            available = pool[~np.isin(pool, contacts)]
            contacts.extend(
                rng.choice(
                    available,
                    size=remaining,
                    replace=len(available) < remaining,
                ).tolist()
            )
        self.network[i] = contacts[:self.network_k]

        # Replace inbound links while approximately preserving the sender's
        # homophily treatment rather than rewiring those links completely at random.
        rows, cols = np.where(self.network == i)
        for r, c in zip(rows, cols):
            candidates = all_idx[(all_idx != r) & (all_idx != i)]
            choose_pool = candidates
            if target_homophily > 0 and rng.random() < target_homophily:
                same_home_r = candidates[home[candidates] == home[r]]
                if len(same_home_r):
                    choose_pool = same_home_r
            self.network[r, c] = int(rng.choice(choose_pool))

    def _demography_step(self, t):
        a = self.agents
        rng = self._rng(61, t)
        factors = self._shock_factors(t)
        age = a["age"].to_numpy(dtype=int)
        mortality = np.clip(mortality_probability(age) * factors["mortality_multiplier"], 0, 0.20)
        deaths = rng.random(len(a)) < mortality
        next_age = age + 1
        age_out = (next_age > 35) & (~deaths)
        outgoing = np.where(deaths | age_out)[0]
        survivor = ~(deaths | age_out)
        a.loc[survivor, "age"] = next_age[survivor]

        intl_entries = 0
        domestic_entries = 0
        if len(outgoing):
            probs = self.base_state["young_adult_population_est_18_35"].to_numpy(dtype=float)
            probs /= probs.sum()
            shuffled = rng.permutation(outgoing)
            expected_intl = self.params["international_entry_rate"] * self.n_agents
            n_intl = min(len(outgoing), int(rng.poisson(expected_intl)))
            intl_idx = shuffled[:n_intl]
            dom_idx = shuffled[n_intl:]
            intl_entries = len(intl_idx)
            domestic_entries = len(dom_idx)

            if len(dom_idx):
                states = rng.choice(STATE_CODES, size=len(dom_idx), p=probs)
                homes = states.copy() if self.entrant_home_local else rng.choice(STATE_CODES, size=len(dom_idx), p=probs)
                a.loc[dom_idx, "age"] = 18
                a.loc[dom_idx, "state"] = states
                a.loc[dom_idx, "home_state"] = homes
                a.loc[dom_idx, "is_international"] = 0

            if len(intl_idx):
                attract = probs * self.state["job_index"].to_numpy() / np.maximum(self.state["rent_index"].to_numpy(), 0.5)
                attract /= attract.sum()
                states = rng.choice(STATE_CODES, size=len(intl_idx), p=attract)
                homes = states.copy() if self.entrant_home_local else rng.choice(STATE_CODES, size=len(intl_idx), p=probs)
                a.loc[intl_idx, "age"] = np.clip(np.rint(rng.normal(27, 4.5, len(intl_idx))), 18, 35).astype(int)
                a.loc[intl_idx, "state"] = states
                a.loc[intl_idx, "home_state"] = homes
                a.loc[intl_idx, "is_international"] = 1

            entrant_states = a.loc[outgoing, "state"].to_numpy()
            entrant_ages = a.loc[outgoing, "age"].to_numpy(dtype=int)
            college_prob = np.array([self.base_state.loc[s, "college_share_proxy"] for s in entrant_states])
            college_prob = np.where(entrant_ages < 22, college_prob * 0.20, np.where(entrant_ages < 25, college_prob * 0.60, college_prob))
            college = rng.binomial(1, np.clip(college_prob, 0.03, 0.75))
            skill = np.clip(rng.normal(1.0 + 0.18 * college, 0.15, len(outgoing)), 0.55, 1.65)
            anchor = np.array([0.65 * self.base_state.loc[s, "median_household_income"] for s in entrant_states])

            a.loc[outgoing, "college"] = college
            a.loc[outgoing, "skill"] = skill
            a.loc[outgoing, "income"] = anchor * (0.72 + 0.28 * skill) * (1 + 0.22 * college) * rng.lognormal(0, 0.12, len(outgoing))
            a.loc[outgoing, "employed"] = 0
            a.loc[outgoing, "has_children"] = 0
            a.loc[outgoing, "dependent_children"] = 0
            a.loc[outgoing, "unemployed_years"] = 0
            a.loc[outgoing, "job_rejected_last_year"] = 0
            a.loc[outgoing, "last_move_year"] = -99

            for i in outgoing:
                a.loc[i, "household_id"] = self.next_household_id
                self.next_household_id += 1
                self._rewire_agent(int(i), t)

        self.demography_rows.append({
            "year": BASE_YEAR + t + 1,
            "deaths": int(deaths.sum()),
            "age_outs": int(age_out.sum()),
            "domestic_entries": int(domestic_entries),
            "international_entries": int(intl_entries),
        })

    def _social_components(self, members, destination, state_idx, home_idx, group_counts, group_totals):
        peer = np.mean([
            np.mean(state_idx[self.network[i]] == destination)
            for i in members
        ])
        coorigin = np.mean([
            group_counts[home_idx[i], destination] / max(group_totals[home_idx[i]], 1)
            for i in members
        ])
        home = np.mean([home_idx[i] == destination for i in members])
        attachment = (
            SOCIAL_UTILITY_WEIGHTS["peer"] * peer
            + SOCIAL_UTILITY_WEIGHTS["coorigin"] * coorigin
            + SOCIAL_UTILITY_WEIGHTS["home"] * home
        ) / (
            SOCIAL_UTILITY_WEIGHTS["peer"]
            + SOCIAL_UTILITY_WEIGHTS["coorigin"]
            + SOCIAL_UTILITY_WEIGHTS["home"]
        )
        return float(peer), float(coorigin), float(home), float(attachment)

    def _base_utility_matrix(self, t):
        a = self.agents
        n = len(a)
        S = len(STATE_CODES)
        current = self._state_indices()
        home = self._home_indices()
        factors = self._shock_factors(t)

        wage = self.state["wage_index"].to_numpy() * factors["wage_multiplier"]
        jobs = self.state["job_index"].to_numpy()
        rent = self.state["rent_index"].to_numpy()
        income_anchor = 0.65 * self.base_state["median_household_income"].to_numpy() * (wage / self.base_state["wage_index"].to_numpy())
        expected_income = (
            income_anchor[None, :]
            * (0.72 + 0.28 * a["skill"].to_numpy()[:, None])
            * (1.0 + 0.22 * a["college"].to_numpy()[:, None])
        )

        U = (
            self.params["beta_income"] * np.log(np.maximum(expected_income, 1000) / np.median(income_anchor))
            + self.params["beta_jobs"] * jobs[None, :]
            - self.params["beta_rent"] * rent[None, :]
            + DESTINATION_EFFECTS.to_numpy()[None, :]
            + self.params["beta_housing_signal"] * self.policies["housing_policy"].to_numpy()[None, :]
            + self.params["beta_employment_signal"] * self.policies["employment_incentive"].to_numpy()[None, :]
            + self.params["beta_migration_incentive"] * self.policies["migration_incentive"].to_numpy()[None, :]
        )
        U -= self.params["beta_distance"] * distance_index.to_numpy()[current, :]
        U -= (
            self.params["migration_cost"]
            * factors["migration_cost_multiplier"]
            * (np.arange(S)[None, :] != current[:, None])
        )

        # Optional crowding cost relative to each state's initial representative population.
        if self.params.get("beta_crowding", 0.0) > 0:
            crowding = np.maximum(self._counts().to_numpy() / np.maximum(self.initial_counts.to_numpy(), 1) - 1.0, 0.0)
            U -= self.params["beta_crowding"] * crowding[None, :]

        U[np.arange(n), current] += (
            self.params["beta_current_state"]
            - self.params["beta_unemployment_push"] * np.minimum(a["unemployed_years"].to_numpy(), 3)
            - self.params["beta_job_rejection_push"] * a["job_rejected_last_year"].to_numpy()
        )
        U[np.arange(n), home] += self.params["beta_home_state"]
        return U, current, home

    def _migration_step(self, t):
        a = self.agents
        S = len(STATE_CODES)
        base_U, state_idx, home_idx = self._base_utility_matrix(t)
        old_states = a["state"].to_numpy().copy()
        state_idx = state_idx.copy()

        group_counts = np.zeros((S, S), dtype=int)
        np.add.at(group_counts, (home_idx, state_idx), 1)
        group_totals = group_counts.sum(axis=1)

        household_counts = a.groupby("state")["household_id"].nunique().reindex(STATE_CODES, fill_value=0).to_numpy()
        housing_slots = np.maximum(
            1,
            np.round(
                household_counts * self.base_state["vacancy_rate"].to_numpy() * self.params["housing_entry_factor"]
                + household_counts * self.policies["housing_policy"].to_numpy() * self.params["housing_policy_slots"]
            ),
        ).astype(int)
        housing_capacity = housing_slots.copy()

        households = list(a.groupby("household_id", sort=False).indices.items())
        rng = self._rng(11, t)
        order = rng.permutation(len(households))

        proposed = rejected = 0
        proposed_by_state = np.zeros(S, dtype=int)
        rejected_by_state = np.zeros(S, dtype=int)
        move_rows = []
        # Target proposal/rejection counts are household-level because housing
        # capacity is allocated to households. Accepted inflow, chain counts,
        # and mover social outcomes are person-level to match agent-year rates.
        target_proposed = target_rejected = 0
        target_accepted_agents = 0
        chain_count_agents = 0
        social_cost_sum = social_cost_n = 0
        origin_attachment_sum = 0.0

        for position in order:
            household_id, members = households[position]
            members = np.asarray(members, dtype=int)
            origin = int(state_idx[members[0]])
            hu = base_U[members].mean(axis=0)

            peer_pull = np.zeros(S)
            co_pull = np.zeros(S)
            for i in members:
                contacts = state_idx[self.network[i]]
                peer_pull += np.bincount(contacts, minlength=S) / self.network_k
                co_pull += group_counts[home_idx[i]] / max(group_totals[home_idx[i]], 1)
            hu += self.params["beta_peer_network"] * peer_pull / len(members)
            hu += self.params["beta_coorigin"] * co_pull / len(members)

            origin_peer, origin_co, origin_home, origin_attach = self._social_components(
                members, origin, state_idx, home_idx, group_counts, group_totals
            )
            dissatisfaction = max(0.0, self.params["social_satisfaction_threshold"] - origin_peer)
            hu[origin] -= self.params["beta_social_dissatisfaction"] * dissatisfaction

            z = hu / self.params["temperature"]
            z -= z.max()
            prob = np.exp(z)
            prob /= prob.sum()
            dest = int(rng.choice(S, p=prob))

            dest_peer, dest_co, dest_home, dest_attach = self._social_components(
                members, dest, state_idx, home_idx, group_counts, group_totals
            )
            social_cost = origin_attach - dest_attach
            recent_contact = False
            contact_ids = np.unique(np.concatenate([self.network[i] for i in members]))
            if len(contact_ids):
                recent_contact = bool(np.any(
                    (state_idx[contact_ids] == dest)
                    & (a.loc[contact_ids, "last_move_year"].to_numpy() >= max(0, t - 2))
                    & (a.loc[contact_ids, "last_move_year"].to_numpy() >= 0)
                ))

            is_move = dest != origin
            accepted = False
            housing_rejected = False

            if is_move:
                proposed += 1
                proposed_by_state[dest] += 1
                if STATE_CODES[dest] == parameters.FOCAL_STATE:
                    target_proposed += 1

                # Smooth housing feasibility rather than a deterministic hard wall.
                # When capacity is plentiful, acceptance is almost certain; near or
                # beyond nominal capacity, probability falls smoothly. Rejections then
                # raise next-year rent through _environment_step.
                nominal_capacity = max(float(housing_capacity[dest]), 1.0)
                remaining_ratio = float(housing_slots[dest]) / nominal_capacity
                softness = max(float(self.params.get("housing_market_softness", 0.10)), 1e-4)
                buffer = float(self.params.get("housing_capacity_buffer", 0.05))
                housing_accept_prob = float(logistic((remaining_ratio - buffer) / softness))

                if rng.random() > housing_accept_prob:
                    rejected += 1
                    rejected_by_state[dest] += 1
                    housing_rejected = True
                    if STATE_CODES[dest] == parameters.FOCAL_STATE:
                        target_rejected += 1
                else:
                    accepted = True
                    housing_slots[dest] -= 1
                    housing_slots[origin] += 1
                    if STATE_CODES[dest] == parameters.FOCAL_STATE:
                        n_people = len(members)
                        target_accepted_agents += n_people
                        chain_count_agents += int(recent_contact) * n_people
                        social_cost_sum += social_cost * n_people
                        origin_attachment_sum += origin_attach * n_people
                        social_cost_n += n_people

                    for i in members:
                        group_counts[home_idx[i], state_idx[i]] -= 1
                        group_counts[home_idx[i], dest] += 1
                        state_idx[i] = dest
                        a.loc[i, "last_move_year"] = t
                        if self.record_events:
                            move_rows.append({
                                "year": BASE_YEAR + t + 1,
                                "agent_id": int(a.loc[i, "agent_id"]),
                                "household_id": int(a.loc[i, "household_id"]),
                                "origin": STATE_CODES[origin],
                                "destination": STATE_CODES[dest],
                                "home_state": a.loc[i, "home_state"],
                                "origin_peer_share": origin_peer,
                                "origin_coorigin_share": origin_co,
                                "destination_peer_share": dest_peer,
                                "destination_coorigin_share": dest_co,
                                "origin_social_attachment": origin_attach,
                                "destination_social_attachment": dest_attach,
                                "social_cost": social_cost,
                                "chain_move": int(recent_contact),
                            })

            if self.record_events:
                self.decision_rows.append({
                    "year": BASE_YEAR + t + 1,
                    "household_id": int(household_id),
                    "origin": STATE_CODES[origin],
                    "chosen_destination": STATE_CODES[dest],
                    "proposed_move": int(is_move),
                    "accepted_move": int(is_move and accepted),
                    "housing_rejected": int(housing_rejected),
                    "origin_peer_share": origin_peer,
                    "origin_coorigin_share": origin_co,
                    "origin_social_attachment": origin_attach,
                    "destination_social_attachment": dest_attach,
                    "social_cost": social_cost,
                    "chain_move": int(recent_contact and is_move and accepted),
                })

        new_states = np.array(STATE_CODES, dtype=object)[state_idx]
        a["state"] = new_states
        if self.record_events and move_rows:
            self.move_history.append(pd.DataFrame(move_rows))

        self.housing_rejection_by_state[t] = {
            state: (rejected_by_state[j] / max(proposed_by_state[j], 1))
            for j, state in enumerate(STATE_CODES)
        }

        self.mechanism_rows.append({
            "year": BASE_YEAR + t + 1,
            "target_proposed": target_proposed,
            "target_accepted": target_accepted_agents,
            "target_housing_rejected": target_rejected,
            "target_chain_moves": chain_count_agents,
            "target_social_cost_sum": social_cost_sum,
            "target_origin_attachment_sum": origin_attachment_sum,
            "target_social_cost_n": social_cost_n,
        })

        return old_states, new_states, proposed, rejected

    def _employment_step(self, t):
        a = self.agents
        n = len(a)
        rng = self._rng(37, t)
        factors = self._shock_factors(t)
        employed = np.zeros(n, dtype=int)
        rejected_flag = np.zeros(n, dtype=int)
        states = a["state"].to_numpy()
        state_rejection = {}
        state_unfilled = {}

        for state in STATE_CODES:
            mask = states == state
            applicants = np.where(mask)[0]
            if len(applicants) == 0:
                state_rejection[state] = np.nan
                state_unfilled[state] = 0
                continue

            slot_rate = np.clip(
                0.80
                + 0.16 * (self.state.loc[state, "job_index"] - 1.0)
                + self.params["employment_policy_slots"] * self.policies.loc[state, "employment_incentive"],
                0.60,
                1.20,
            )
            slots = int(round(slot_rate * mask.sum() * factors["job_slot_multiplier"]))
            order = rng.permutation(applicants)
            remaining = slots
            for i in order:
                if remaining <= 0:
                    break
                p_match = logistic(
                    2.0
                    + 0.8 * (a.loc[i, "skill"] - 1.0)
                    + 0.30 * a.loc[i, "college"]
                    - 0.10 * min(a.loc[i, "unemployed_years"], 3)
                )
                if rng.random() < p_match:
                    employed[i] = 1
                    remaining -= 1

            rejected_here = applicants[employed[applicants] == 0]
            rejected_flag[rejected_here] = 1
            state_rejection[state] = len(rejected_here) / max(len(applicants), 1)
            state_unfilled[state] = max(remaining, 0)

        self.job_rejection_by_state[t] = state_rejection
        self.unfilled_jobs_by_state[t] = state_unfilled
        a["employed"] = employed
        a["job_rejected_last_year"] = rejected_flag
        a["unemployed_years"] = np.where(employed == 1, 0, a["unemployed_years"].to_numpy() + 1)

        wage = a["state"].map(self.state["wage_index"]).to_numpy() * factors["wage_multiplier"]
        base_wage = a["state"].map(self.base_state["wage_index"]).to_numpy()
        anchor = 0.65 * a["state"].map(self.base_state["median_household_income"]).to_numpy()
        income = (
            anchor * (wage / base_wage)
            * (0.72 + 0.28 * a["skill"].to_numpy())
            * (1.0 + 0.22 * a["college"].to_numpy())
            * np.where(employed == 1, 1.0, 0.34)
        )
        a["income"] = income * self._rng(41, t).lognormal(0, 0.04, n)

    def _environment_step(self):
        current = self._counts()
        growth = (current - self.previous_counts) / np.maximum(self.previous_counts, 1.0)

        # State-specific excess housing demand is translated into next-year rent
        # pressure. Current-year feasibility is soft/probabilistic near capacity,
        # while this lagged price channel dampens subsequent demand.
        if self.housing_rejection_by_state:
            latest_t = max(self.housing_rejection_by_state)
            rejection_rate = pd.Series(
                self.housing_rejection_by_state[latest_t],
                index=STATE_CODES,
                dtype=float,
            ).fillna(0.0)
        else:
            rejection_rate = pd.Series(0.0, index=STATE_CODES)
        rent_pressure = self.params.get("rent_rejection_elasticity", 0.0) * rejection_rate

        rent_growth = (
            0.10 * growth
            - self.params["housing_policy_effect"] * self.policies["housing_policy"]
            + 0.005 * (1.0 / self.state["housing_capacity_index"] - 1.0)
            + rent_pressure
        ).clip(-0.04, 0.12)
        self.state["rent_index"] = (self.state["rent_index"] * (1 + rent_growth)).clip(0.70, 1.80)

        skill = self.agents.groupby("state")["skill"].mean().reindex(STATE_CODES).fillna(1.0)
        job_growth = (
            0.008 * (skill - 1.0)
            + 0.008 * growth.clip(-0.15, 0.15)
            + self.params["employment_policy_growth"] * self.policies["employment_incentive"]
        ).clip(-0.04, 0.06)
        self.state["job_index"] = (self.state["job_index"] * (1 + job_growth)).clip(0.75, 1.40)
        self.previous_counts = current

    def _agent_social_utility_components(self):
        a = self.agents
        state_idx = self._state_indices()
        home_idx = self._home_indices()
        contact_state = state_idx[self.network]
        peer = (contact_state == state_idx[:, None]).mean(axis=1)

        S = len(STATE_CODES)
        counts = np.zeros((S, S), dtype=int)
        np.add.at(counts, (home_idx, state_idx), 1)
        totals = counts.sum(axis=1)
        co = np.zeros(len(a), dtype=float)
        valid = totals[home_idx] > 1
        co[valid] = (counts[home_idx[valid], state_idx[valid]] - 1) / (totals[home_idx[valid]] - 1)
        home = (state_idx == home_idx).astype(float)

        monthly = self.base_state["median_gross_rent_monthly"] * (self.state["rent_index"] / self.base_state["rent_index"])
        rent_burden = a["state"].map(monthly).to_numpy() * 12 / np.maximum(a["income"].to_numpy(), 12000)
        affordability = 1 - np.clip(rent_burden / 0.60, 0, 1)
        income_rel = a["income"].to_numpy() / max(a["income"].median(), 1)
        income_score = logistic(np.log(np.maximum(income_rel, 0.05)))

        total = (
            SOCIAL_UTILITY_WEIGHTS["peer"] * peer
            + SOCIAL_UTILITY_WEIGHTS["coorigin"] * co
            + SOCIAL_UTILITY_WEIGHTS["home"] * home
            + SOCIAL_UTILITY_WEIGHTS["employment"] * a["employed"].to_numpy()
            + SOCIAL_UTILITY_WEIGHTS["income"] * income_score
            + SOCIAL_UTILITY_WEIGHTS["affordability"] * affordability
        )
        return peer, co, home, rent_burden, total

    def _record_metrics(self, t, old_states, new_states, proposed, rejected, births):
        peer, co, home, rent_burden, social_utility = self._agent_social_utility_components()
        current_states = self.agents["state"].to_numpy()
        pop_share = self._counts() / len(self.agents)
        system = {
            "peer_colocation": float(np.mean(peer)),
            "coorigin_clustering": float(np.mean(co)),
            "population_hhi": float((pop_share ** 2).sum()),
            "move_rate": float(np.mean(old_states != new_states)),
            "housing_rejection_rate": rejected / max(proposed, 1),
            "mean_social_utility_index": float(np.mean(social_utility)),
            "mean_rent_burden_system": float(np.mean(rent_burden)),
            "births": int(births),
            "unfilled_jobs": int(sum(self.unfilled_jobs_by_state.get(t, {}).values())),
        }
        for state in STATE_CODES:
            mask = current_states == state
            g = self.agents.loc[mask]
            inflow = int(((old_states != state) & (new_states == state)).sum())
            outflow = int(((old_states == state) & (new_states != state)).sum())
            self.history.append({
                "year": BASE_YEAR + t + 1,
                "state": state,
                "population": len(g),
                "inflow": inflow,
                "outflow": outflow,
                "net_migration": inflow - outflow,
                "employment_rate": g["employed"].mean() if len(g) else np.nan,
                "mean_income": g["income"].mean() if len(g) else np.nan,
                "rent_index": self.state.loc[state, "rent_index"],
                "job_rejection_rate_state": self.job_rejection_by_state.get(t, {}).get(state, np.nan),
                "unfilled_jobs_state": self.unfilled_jobs_by_state.get(t, {}).get(state, 0),
                **system,
            })

    def _snapshot(self, year):
        self.snapshots[year] = self.agents[[
            "agent_id", "state", "home_state", "age", "employed",
            "household_id", "has_children", "is_international"
        ]].copy()

    def step(self, t):
        self._demography_step(t)
        self._household_formation_step(t)
        births = self._family_step(t)
        old_states, new_states, proposed, rejected = self._migration_step(t)
        self._employment_step(t)
        self._environment_step()
        self._record_metrics(t, old_states, new_states, proposed, rejected, births)
        if t + 1 in {5, 10, 15, 20}:
            self._snapshot(BASE_YEAR + t + 1)

    def run(self):
        for t in range(self.years):
            self.step(t)
        history = pd.DataFrame(self.history)
        moves = (
            pd.concat(self.move_history, ignore_index=True)
            if self.move_history
            else pd.DataFrame(columns=["year", "agent_id", "origin", "destination"])
        )
        return history, moves, self.snapshots

    def mechanism_history(self):
        return pd.DataFrame(self.mechanism_rows)

    def decision_history(self):
        return pd.DataFrame(self.decision_rows)
