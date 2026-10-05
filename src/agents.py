"""Agent population and initial social network (Section 5)."""
import numpy as np
import pandas as pd

from src.config import STATE_CODES
from utils.stats import logistic


def initialise_agents(
    state_df,
    n_agents,
    seed,
    network_k=8,
    couple_share=0.35,
    network_homophily_share=0.0,
):
    rng = np.random.default_rng(seed)
    probs = state_df["young_adult_population_est_18_35"].to_numpy(dtype=float)
    probs /= probs.sum()

    states = rng.choice(STATE_CODES, size=n_agents, p=probs)
    home_states = rng.choice(STATE_CODES, size=n_agents, p=probs)
    age = rng.integers(18, 36, size=n_agents)

    college_prob = np.array([state_df.loc[s, "college_share_proxy"] for s in states])
    college_prob = np.where(age < 22, college_prob * 0.20, np.where(age < 25, college_prob * 0.60, college_prob))
    college = rng.binomial(1, np.clip(college_prob, 0.03, 0.75))
    skill = np.clip(rng.normal(1.0 + 0.18 * college, 0.15, n_agents), 0.55, 1.65)

    has_children = rng.binomial(1, logistic((age - 28) / 3.5 - 1))
    dependent_children = has_children * np.clip(rng.poisson(1.0, n_agents) + 1, 1, 3)

    income_anchor = np.array([0.65 * state_df.loc[s, "median_household_income"] for s in states])
    income = (
        income_anchor
        * (0.72 + 0.28 * skill)
        * (1.0 + 0.22 * college)
        * rng.lognormal(0.0, 0.12, n_agents)
    )

    agents = pd.DataFrame({
        "agent_id": np.arange(n_agents),
        "age": age,
        "college": college,
        "skill": skill,
        "income": income,
        "employed": np.ones(n_agents, dtype=int),
        "has_children": has_children,
        "dependent_children": dependent_children,
        "home_state": home_states,
        "state": states,
        "is_international": np.zeros(n_agents, dtype=int),
        "unemployed_years": np.zeros(n_agents, dtype=int),
        "job_rejected_last_year": np.zeros(n_agents, dtype=int),
        "last_move_year": np.full(n_agents, -99, dtype=int),
    })

    household_id = np.arange(n_agents, dtype=int)
    next_household = n_agents
    for state in STATE_CODES:
        eligible = np.where((states == state) & (age >= 22))[0]
        rng.shuffle(eligible)
        n_pairs = int((len(eligible) * couple_share) // 2)
        for a, b in eligible[: 2 * n_pairs].reshape(-1, 2):
            household_id[a] = next_household
            household_id[b] = next_household
            dep = max(int(agents.loc[a, "dependent_children"]), int(agents.loc[b, "dependent_children"]))
            if dep > 0:
                agents.loc[[a, b], "has_children"] = 1
                agents.loc[[a, b], "dependent_children"] = dep
            next_household += 1
    agents["household_id"] = household_id

    # Social-network initialization. The baseline remains random (share=0), while
    # the robustness experiment sets share=0.5 to introduce realistic co-origin
    # homophily without forcing all contacts to be co-origin.
    homophily = float(np.clip(network_homophily_share, 0.0, 1.0))
    network = np.empty((n_agents, network_k), dtype=int)
    all_idx = np.arange(n_agents)
    home_arr = agents["home_state"].to_numpy()

    for i in range(n_agents):
        pool = all_idx[all_idx != i]
        contacts = []
        k_home = int(round(network_k * homophily))
        if k_home > 0:
            same_home_pool = pool[home_arr[pool] == home_arr[i]]
            if len(same_home_pool):
                take = min(k_home, len(same_home_pool))
                contacts.extend(rng.choice(same_home_pool, size=take, replace=False).tolist())

        remaining = network_k - len(contacts)
        if remaining > 0:
            available = pool[~np.isin(pool, contacts)]
            contacts.extend(
                rng.choice(
                    available,
                    size=remaining,
                    replace=len(available) < remaining,
                ).tolist()
            )
        network[i] = contacts[:network_k]

    return agents, network
