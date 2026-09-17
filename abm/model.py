"""Agent-based model of young-adult interstate migration in the United States.

Each agent is one young adult aged 18-34, carrying four attributes:

    age          years
    education    0 = no bachelor's degree, 1 = bachelor's or above
    income       annual earnings, USD
    rent_burden  annual rent in the agent's state divided by that income

plus a location (the state they currently live in).

Every year each agent draws a small set of candidate destination states and picks
where to live by random-utility (multinomial logit) choice. Aggregate migration
then feeds back into state rents and wages, which changes next year's decisions.

State initial conditions come from the Census ACS (see data/fetch_census.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import geography as geo

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@dataclass
class Params:
    """Model parameters. Utility weights are in arbitrary logit units."""

    n_agents: int = 30_000
    years: int = 20
    seed: int = 0

    # --- migration utility weights -------------------------------------------
    beta_income: float = 2.6      # pull of higher expected earnings
    beta_rent: float = 3.4        # push of a higher rent burden
    beta_edu: float = 1.2         # pull of college access, for those still seeking it
    beta_distance: float = 0.75   # cost of moving further
    beta_home: float = 5.6        # inertia: staying put is the default
    n_candidates: int = 6         # destinations an agent actually considers

    # How geography enters the migration decision. The three settings are
    # alternative assumptions about what "distance" means, compared in the report:
    #   "centroid"  - cost depends on centroid-to-centroid distance alone.
    #   "border"    - subtract the origin state's own radius, so cost is distance
    #                 beyond getting out of your own state.
    #   "size_home" - keep centroid distance, but scale the stay-put bonus by the
    #                 origin's radius: more of a large state's opportunities lie
    #                 inside it, so leaving is a bigger step.
    # Both alternatives are derived from Census land area and add no fitted
    # parameter; "size_home" is renormalised so the mean bonus is unchanged.
    geography_mode: str = "centroid"
    min_effective_distance: float = 40.0  # km floor, guards against negatives

    # --- macro feedback ------------------------------------------------------
    rent_elasticity: float = 0.85     # rent response to demand/supply imbalance
    # Baseline housing growth is zero because the national cohort size is held
    # fixed: the model studies how young adults redistribute, not how many there
    # are. Any positive value would make supply outrun demand nationally and drive
    # rents down everywhere, swamping the between-state differences we care about.
    housing_growth: float = 0.0
    wage_elasticity: float = 0.30     # wage response to skill mix
    agglomeration: float = 0.45       # college density raises local wages
    congestion: float = 0.35          # labour supply growth dampens wages
    income_growth: float = 0.012      # baseline real earnings growth

    # --- individual life course ----------------------------------------------
    age_min: int = 18
    age_max: int = 34
    edu_entry_age: int = 24           # last age at which college entry is modelled
    income_age_premium: float = 0.028 # annual earnings growth with experience


@dataclass
class Policy:
    """A state-government intervention, held at a fixed annual budget.

    Each lever is a per-state intensity in [0, 1]. `budget_per_state` scales how
    much effect a unit of intensity buys, so that scenarios can be compared at
    equal fiscal cost.
    """

    education: dict[str, float] = field(default_factory=dict)
    housing: dict[str, float] = field(default_factory=dict)
    employment: dict[str, float] = field(default_factory=dict)
    start_year: int = 1

    def total_intensity(self) -> float:
        return sum(
            sum(lever.values())
            for lever in (self.education, self.housing, self.employment)
        )

    def as_arrays(self, order: list[str]) -> dict[str, np.ndarray]:
        out = {}
        for name in ("education", "housing", "employment"):
            lever = getattr(self, name)
            out[name] = np.array([lever.get(s, 0.0) for s in order], dtype=float)
        return out


class MigrationModel:
    def __init__(
        self,
        params: Params | None = None,
        policy: Policy | None = None,
        states_csv: Path | str | None = None,
    ):
        self.p = params or Params()
        self.policy = policy or Policy()
        self.rng = np.random.default_rng(self.p.seed)
        self.states_csv = Path(states_csv) if states_csv else DATA / "states_acs.csv"

        self._load_states()
        self._init_state_variables()
        self._init_agents()

        self.history: list[dict] = []
        self.year_flow_history: list[np.ndarray] = []
        self.flow_matrix = np.zeros((self.n_states, self.n_states), dtype=np.int64)
        self.year = 0
        self._record()

    # ------------------------------------------------------------------ setup

    def _load_states(self) -> None:
        path = self.states_csv
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run: python data/fetch_census.py\n"
                "(needs a free Census API key)"
            )
        df = pd.read_csv(path, dtype={"state_fips": str})
        df["state_fips"] = df["state_fips"].str.zfill(2)

        abbr = geo.fips_to_abbr()
        df["abbr"] = df["state_fips"].map(abbr)
        df = df[df["abbr"].notna()].sort_values("abbr").reset_index(drop=True)

        self.states = df
        self.order: list[str] = df["abbr"].tolist()
        self.n_states = len(self.order)
        self.state_index = {a: i for i, a in enumerate(self.order)}

        self.adjacent = geo.adjacency_matrix(self.order)
        self.radius = geo.radius_array(self.order)

        mode = self.p.geography_mode
        if mode not in ("centroid", "border", "size_home"):
            raise ValueError(f"unknown geography_mode: {mode!r}")

        self.distance = geo.distance_matrix(self.order)
        if mode == "border":
            self.distance = np.maximum(
                self.distance - self.radius[:, None], self.p.min_effective_distance
            )

        # Per-origin stay-put bonus. Only "size_home" varies it by state; the mean
        # is held at beta_home so the calibrated national move rate still applies.
        if mode == "size_home":
            weight = np.log(self.radius)
            self.home_bonus = self.p.beta_home * weight / weight.mean()
        else:
            self.home_bonus = np.full(self.n_states, self.p.beta_home)

    def _init_state_variables(self) -> None:
        df = self.states
        p = self.p

        # Annual earnings by education. ACS medians are for workers 25+, so the
        # non-college wage blends high-school and some-college earners.
        self.wage = np.zeros((2, self.n_states))
        self.wage[0] = df[["earn_median_hs", "earn_median_somecollege"]].mean(axis=1)
        self.wage[1] = df[["earn_median_bachelors", "earn_median_graduate"]].mean(axis=1)
        self.wage = np.nan_to_num(self.wage, nan=np.nanmean(self.wage))
        self.wage0 = self.wage.copy()
        # The exogenous earnings trend, before any agglomeration premium.
        self.wage_trend = self.wage.copy()

        self.rent = df["median_gross_rent"].to_numpy(dtype=float) * 12.0
        self.rent = np.nan_to_num(self.rent, nan=np.nanmean(self.rent))
        self.rent0 = self.rent.copy()

        # College accessibility, anchored on each state's observed degree share.
        share = df["bachelors_plus_share"].to_numpy(dtype=float)
        self.edu_access = np.clip(share / share.max(), 0.15, 1.0)
        self.edu_access0 = self.edu_access.copy()

        self.unemployment = np.nan_to_num(
            df["unemployment_rate"].to_numpy(dtype=float), nan=0.05
        )

        # One simulated agent stands for this many real young adults.
        self.pop_18_34 = df["pop_18_34"].to_numpy(dtype=float)
        self.scale = self.pop_18_34.sum() / p.n_agents

        self.housing_stock = self.pop_18_34.copy()
        self.birth_shares = self.pop_18_34 / self.pop_18_34.sum()

        # Gravity prior over destinations: bigger and closer states get considered
        # more often. This is the agent's awareness set, not their choice.
        d = self.distance.copy()
        np.fill_diagonal(d, np.inf)
        grav = self.pop_18_34[None, :] / np.power(1.0 + d, 1.5)
        self.gravity_cdf = np.cumsum(grav / grav.sum(axis=1, keepdims=True), axis=1)

    def _init_agents(self) -> None:
        p = self.p
        rng = self.rng
        n = p.n_agents

        self.a_state = rng.choice(self.n_states, size=n, p=self.birth_shares)
        self.a_age = rng.integers(p.age_min, p.age_max + 1, size=n)

        share = self.states["bachelors_plus_share"].to_numpy(dtype=float)
        eligible = self.a_age >= 22
        self.a_edu = (rng.random(n) < share[self.a_state]).astype(np.int8)
        self.a_edu[~eligible] = 0

        self.a_income = self._draw_income(self.a_state, self.a_edu, self.a_age)
        self.a_rent_burden = self.rent[self.a_state] / self.a_income

        # Reference graduate share, so the agglomeration premium is measured as a
        # deviation from where each state actually started.
        counts = np.bincount(self.a_state, minlength=self.n_states).astype(float)
        college = np.bincount(
            self.a_state, weights=self.a_edu.astype(float), minlength=self.n_states
        )
        self.college_share0 = np.divide(
            college, counts, out=np.full(self.n_states, 0.3), where=counts > 0
        )

    def _draw_income(self, state, edu, age) -> np.ndarray:
        """Lognormal earnings around the state-by-education median, aged down."""
        base = self.wage[edu, state]
        # Young workers earn below the 25+ median; the gap closes with experience.
        experience = np.clip(age - self.p.age_min, 0, None)
        age_factor = 0.62 * np.exp(self.p.income_age_premium * experience * 1.9)
        noise = np.exp(self.rng.normal(0.0, 0.42, size=len(state)))
        return np.maximum(base * age_factor * noise, 9_000.0)

    # ------------------------------------------------------------------- step

    def step(self) -> None:
        self.year += 1
        self._age_and_replace()
        self._education_transition()
        self._income_update()
        self._migrate()
        self._macro_update()
        self._record()

    def run(self, years: int | None = None) -> pd.DataFrame:
        for _ in range(years or self.p.years):
            self.step()
        return self.results()

    def _policy_now(self) -> dict[str, np.ndarray]:
        arrays = self.policy.as_arrays(self.order)
        if self.year < self.policy.start_year:
            return {k: np.zeros_like(v) for k, v in arrays.items()}
        return arrays

    def _age_and_replace(self) -> None:
        p = self.p
        self.a_age += 1
        leaving = self.a_age > p.age_max
        k = int(leaving.sum())
        if k == 0:
            return
        # Agents ageing out of the cohort are replaced by new 18-year-olds, born
        # in proportion to where young people currently live.
        idx = np.flatnonzero(leaving)
        self.a_state[idx] = self.rng.choice(self.n_states, size=k, p=self.birth_shares)
        self.a_age[idx] = p.age_min
        self.a_edu[idx] = 0
        self.a_income[idx] = self._draw_income(
            self.a_state[idx], self.a_edu[idx], self.a_age[idx]
        )

    def _education_transition(self) -> None:
        p = self.p
        pol = self._policy_now()
        seekers = (self.a_edu == 0) & (self.a_age <= p.edu_entry_age)
        if not seekers.any():
            return
        idx = np.flatnonzero(seekers)
        s = self.a_state[idx]
        # Baseline graduation hazard, lifted where the state invests in education.
        prob = 0.11 * self.edu_access[s] * (1.0 + 1.6 * pol["education"][s])
        graduated = idx[self.rng.random(len(idx)) < prob]
        self.a_edu[graduated] = 1
        # Graduating lifts earnings toward the college wage in the current state.
        self.a_income[graduated] *= (
            self.wage[1, self.a_state[graduated]]
            / np.maximum(self.wage[0, self.a_state[graduated]], 1.0)
        ) ** 0.6

    def _income_update(self) -> None:
        p = self.p
        target = self.wage[self.a_edu, self.a_state]
        experience = np.clip(self.a_age - p.age_min, 0, None)
        target = target * (0.62 * np.exp(p.income_age_premium * experience * 1.9))
        # Partial adjustment toward the state-and-education target, plus noise.
        self.a_income = np.maximum(
            0.72 * self.a_income
            + 0.28 * target * np.exp(self.rng.normal(0.0, 0.16, size=len(self.a_income))),
            9_000.0,
        )
        self.a_rent_burden = self.rent[self.a_state] / self.a_income

    def _migrate(self) -> None:
        p = self.p
        rng = self.rng
        pol = self._policy_now()
        n = p.n_agents
        k = p.n_candidates

        origin = self.a_state
        # Sample the awareness set from the gravity prior.
        cdf = self.gravity_cdf[origin]                      # (n, n_states)
        r = rng.random((n, k))
        cand = (r[:, :, None] > cdf[:, None, :]).sum(axis=2)
        cand = np.clip(cand, 0, self.n_states - 1)
        options = np.concatenate([origin[:, None], cand], axis=1)   # (n, k+1)

        edu = self.a_edu[:, None]
        # Agents carry their relative position in the wage distribution with them.
        rel = (self.a_income / np.maximum(self.wage[self.a_edu, origin], 1.0))[:, None]
        expected_income = np.maximum(self.wage[edu, options] * rel, 9_000.0)

        rent_burden = self.rent[options] / expected_income

        seeking_edu = ((self.a_edu == 0) & (self.a_age <= p.edu_entry_age))[:, None]
        edu_pull = self.edu_access[options] * seeking_edu

        dist = self.distance[origin[:, None], options]
        is_home = (options == origin[:, None]).astype(float)

        # Employment policy reads as a higher chance of finding work locally.
        job_pull = pol["employment"][options]

        utility = (
            p.beta_income * np.log(expected_income / 1e4)
            - p.beta_rent * rent_burden
            + p.beta_edu * edu_pull
            - p.beta_distance * np.log1p(dist / 500.0)
            + self.home_bonus[origin][:, None] * is_home
            + 0.9 * job_pull
        )
        # Gumbel noise + argmax is exactly multinomial logit choice.
        utility += rng.gumbel(size=utility.shape)

        chosen = options[np.arange(n), utility.argmax(axis=1)]
        moved = chosen != origin
        self.year_flows = np.zeros((self.n_states, self.n_states), dtype=np.int64)
        np.add.at(self.year_flows, (origin[moved], chosen[moved]), 1)
        self.flow_matrix += self.year_flows
        self.last_moves = (origin[moved].copy(), chosen[moved].copy())
        self.a_state = chosen
        self.a_rent_burden = self.rent[self.a_state] / self.a_income

    def _macro_update(self) -> None:
        p = self.p
        pol = self._policy_now()

        counts = np.bincount(self.a_state, minlength=self.n_states).astype(float)
        college = np.bincount(
            self.a_state, weights=self.a_edu.astype(float), minlength=self.n_states
        )
        college_share = np.divide(
            college, counts, out=np.full(self.n_states, 0.3), where=counts > 0
        )

        # Housing: supply grows only where the state invests (see `housing_growth`).
        self.housing_stock *= 1.0 + p.housing_growth + 0.035 * pol["housing"]
        demand_pressure = counts * self.scale / np.maximum(self.housing_stock, 1.0)

        # Rent and wages are *level* relationships, not growth rates: each is
        # anchored to its observed value and re-derived from current conditions
        # every year. Writing them as compounding updates lets a state that starts
        # slightly ahead accumulate the same premium indefinitely, which produces
        # runaway concentration rather than the equilibrating pressure we want.
        self.rent = self.rent0 * np.power(np.maximum(demand_pressure, 0.05), p.rent_elasticity)

        # The exogenous earnings trend is the only compounding term.
        self.wage_trend *= 1.0 + p.income_growth + 0.030 * pol["employment"]
        # A denser graduate pool raises local productivity; a larger labour force
        # relative to the state's starting size bids wages back down.
        density = college_share - self.college_share0
        supply_ratio = counts * self.scale / np.maximum(self.pop_18_34, 1.0)
        premium = 1.0 + p.agglomeration * density - p.congestion * (supply_ratio - 1.0)
        self.wage = self.wage_trend * np.clip(premium, 0.55, 1.8)

        self.edu_access = np.clip(self.edu_access0 * (1.0 + 0.30 * pol["education"]), 0, 1.2)

    # ---------------------------------------------------------------- outputs

    def _record(self) -> None:
        counts = np.bincount(self.a_state, minlength=self.n_states).astype(float)
        college = np.bincount(
            self.a_state, weights=self.a_edu.astype(float), minlength=self.n_states
        )
        income = np.bincount(
            self.a_state, weights=self.a_income, minlength=self.n_states
        )
        burden = np.bincount(
            self.a_state, weights=self.a_rent_burden, minlength=self.n_states
        )
        safe = np.maximum(counts, 1.0)

        if self.year == 0:
            inflow = outflow = np.zeros(self.n_states)
            year_flows = np.zeros((self.n_states, self.n_states), dtype=np.int64)
        else:
            origin, dest = self.last_moves
            outflow = np.bincount(origin, minlength=self.n_states).astype(float)
            inflow = np.bincount(dest, minlength=self.n_states).astype(float)
            year_flows = self.year_flows

        self.year_flow_history.append(year_flows * self.scale)
        self.history.append(
            {
                "year": self.year,
                "abbr": list(self.order),
                "population": counts * self.scale,
                "college_share": college / safe,
                "mean_income": income / safe,
                "mean_rent_burden": burden / safe,
                "rent": self.rent.copy(),
                "wage_college": self.wage[1].copy(),
                "wage_noncollege": self.wage[0].copy(),
                "inflow": inflow * self.scale,
                "outflow": outflow * self.scale,
                "net_migration": (inflow - outflow) * self.scale,
            }
        )

    def results(self) -> pd.DataFrame:
        frames = []
        for snap in self.history:
            frames.append(pd.DataFrame({k: v for k, v in snap.items() if k != "year"}).assign(year=snap["year"]))
        df = pd.concat(frames, ignore_index=True)
        return df[["year", "abbr"] + [c for c in df.columns if c not in ("year", "abbr")]]

    def annual_flows(self) -> list[np.ndarray]:
        """One (n_states, n_states) matrix per recorded year, scaled to real people."""
        return self.year_flow_history
