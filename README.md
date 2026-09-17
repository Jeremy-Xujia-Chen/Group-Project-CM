# Group-Project-CM

A spatial agent-based model of young-adult interstate migration in the United States,
initialised from US Census ACS data. Agents aged 18–34 choose each year whether to stay
or move; their choices feed back into state rents and wages. The model is used to compare
state policy scenarios at equal budget, not to forecast.

The original research proposal is in [`proposal.md`](proposal.md). The unit's assessment
criteria are in [`specifications.md`](specifications.md).

---

## Running it

### 1. Install

Python 3.10 or newer.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Get a Census API key

Free and instant, from <https://api.census.gov/data/key_signup.html>. Save it to a file
in the project root (already gitignored):

```bash
echo "YOUR_KEY_HERE" > .census_api_key
```

Or export it as `CENSUS_API_KEY` instead.

### 3. Fetch the data

```bash
.venv/bin/python data/fetch_boundaries.py   # state outlines, no key needed
.venv/bin/python data/fetch_census.py       # ACS state table, needs the key
```

Both cache to `data/`, so they only need to run once.

### 4. Run

```bash
.venv/bin/python run_demo.py            # calibrate, validate, map, all scenarios
.venv/bin/python run_demo.py --quick    # smaller and faster, for a live demo
.venv/bin/python compare_assumptions.py # compare alternative distance assumptions
```

Everything lands in `outputs/`. The full run takes a couple of minutes, most of it
rendering the animation.

Useful flags for `run_demo.py`:

| flag | effect |
|---|---|
| `--quick` | 12,000 agents, 12 years, 2 seeds |
| `--agents N` | number of agents (default 40,000) |
| `--years N` | simulation horizon (default 20) |
| `--seeds N` | random seeds per scenario (default 5) |
| `--gif` | write an animated GIF instead of MP4 (no ffmpeg needed) |
| `--no-calibrate` | skip calibration and use the stored `beta_home` |

---

## How it works

### Agents

One agent is one young adult. Each carries four attributes, plus the state they live in:

| attribute | meaning |
|---|---|
| `age` | 18–34; agents ageing out are replaced by new 18-year-olds |
| `education` | 0 = no bachelor's degree, 1 = bachelor's or above |
| `income` | annual earnings in USD |
| `rent_burden` | annual rent in their state, divided by their income |

The attribute set is deliberately small. Everything else — firms, universities, state
governments — is a state-level mechanism rather than a separate agent type, so that
emergent outcomes stay attributable to a legible set of rules.

### One year of the simulation

1. **Age and replace.** Everyone ages a year. Those passing 34 leave the cohort and are
   replaced by new 18-year-olds, born in proportion to where young people currently live.
2. **Education.** Agents under 25 without a degree graduate with a probability set by
   their state's college access, raised where the state invests in education.
3. **Income.** Earnings adjust partway toward the state-and-education median, with an
   experience premium and noise.
4. **Migration.** Each agent draws a handful of candidate destinations from a gravity
   prior (bigger and closer states are considered more often), scores each on expected
   income, rent burden, college access, distance and a stay-put bonus, then picks the
   best after adding Gumbel noise — which makes the choice exactly multinomial logit.
5. **Macro feedback.** State rents rise where population presses on housing stock; wages
   rise where the graduate pool is dense and fall where the labour force has grown.

Rents and wages are written as **level** relationships anchored to their observed ACS
values, re-derived from current conditions each year, rather than as compounding growth
rates. This matters: as compounding updates, a state starting slightly ahead accumulates
the same premium indefinitely, which produced runaway concentration instead of
equilibrating pressure.

### Calibration

The model has one parameter with no direct empirical counterpart: `beta_home`, the
utility bonus for staying put. It is pinned down by bisection, so that the simulated
share of 18–34 year olds who change state each year matches the rate ACS table B07001
observes (about 4.6% nationally).

Everything else is either observed in the data or a structural elasticity. The model's
*spatial* pattern — which states gain and which lose — is therefore a prediction rather
than an input, and `run_demo.py` checks it against the observed per-state in-migration
rates.

### Policy scenarios

Three levers, each a per-state intensity in [0, 1]:

- **education** — raises the college-access rate
- **housing** — raises the growth rate of housing stock, easing rent pressure
- **employment** — raises wages and the pull of local job availability

Every scenario spends the same total budget (8.0 intensity-units nationally), so results
reflect *where and how* money is spent rather than how much:

| scenario | what it does |
|---|---|
| `baseline` | no policy |
| `education` / `housing` / `employment` | one lever, spread evenly across all states |
| `mixed` | all three levers at a third intensity each |
| `housing_concentrated` | the whole housing budget in the 8 largest states |

Each runs across multiple seeds; reported indicators carry a standard deviation.

### Comparing modelling assumptions

`compare_assumptions.py` swaps in three different accounts of what distance means in the
migration decision, recalibrating each so they share the same national move rate and
differ only in *where* people go. Set via `Params.geography_mode`:

- `centroid` — centroid-to-centroid distance alone (default)
- `border` — subtract the origin state's own radius: distance beyond getting out
- `size_home` — scale the stay-put bonus by state size, since more of a large state's
  opportunities lie inside it

Both alternatives derive from Census land area and add no fitted parameter.

---

## What each file does

### Model

| file | role |
|---|---|
| `abm/model.py` | The ABM: agent state, the yearly step, the logit migration choice, macro feedback. `Params` holds every tunable; `Policy` describes an intervention. |
| `abm/geography.py` | State outlines from the Census boundary file, Albers projection with Alaska/Hawaii insets, centroids, great-circle distances, land-area radii, and the state adjacency list. |
| `abm/calibration.py` | Bisects `beta_home` against the observed move rate, and scores simulated per-state in-migration against ACS. |
| `abm/experiments.py` | Builds the equal-budget scenario set, runs them across seeds, and collapses per-state output into national indicators (graduate concentration, rent burden, income dispersion). Also the neighbour-spillover comparison. |
| `abm/viz.py` | The map — states shaded by net migration with curved flow arrows, animated year by year — plus the validation scatter and the scenario panels. |

### Data

| file | role |
|---|---|
| `data/fetch_census.py` | Pulls the ACS 5-year state table: population 18–34, earnings by education, rents, degree share, unemployment, and the observed interstate in-migration rate for the 18–34 cohort. `--flows` additionally builds the state-to-state matrix from county pairs (slow, optional). |
| `data/fetch_boundaries.py` | Downloads and extracts the Census cartographic boundary shapefile. No API key. |
| `data/states_acs.csv` | Cached output of the above — what the model reads at startup. |

### Entry points

| file | role |
|---|---|
| `run_demo.py` | The whole pipeline: calibrate, run the baseline, validate, render the map, run every scenario across seeds, measure spillover. |
| `compare_assumptions.py` | Runs the three distance assumptions side by side with uncertainty across seeds. |

---

## Outputs

Written to `outputs/` (gitignored).

| file | contents |
|---|---|
| `migration.mp4` / `.gif` | The animated map: net migration by state, with flow arrows, year by year |
| `migration_final_year.png` | Still frame of the final year |
| `validation.png` / `.csv` | Simulated vs observed in-migration rate, one point per state |
| `baseline_by_state_year.csv` | Full baseline trajectory: population, income, rent, degree share, flows |
| `scenarios.png` | National indicators over time, one line per scenario, ±1 sd band |
| `scenario_results.csv` | Per-state, per-year, per-seed output for every scenario |
| `scenario_measures.csv` / `scenario_summary.csv` | National indicators, and their final-year summary |
| `spillover.csv` | Population change in treated states, their neighbours, and everywhere else |
| `assumption_comparison.csv` | The distance-assumption comparison |

---

## Known limitations

These are consequences of the model's scope, and are material for the report rather than
defects to patch.

**The baseline does not reproduce observed migration geography.** Texas and Florida come
out as net losers; in reality they are the largest gainers. Nothing in four attributes
represents job growth or climate, which is what actually drives Sun Belt migration.

**Validation is weak, and the headline number flatters it.** Pearson r ≈ 0.46 against
observed per-state in-migration rates, but this is carried almost entirely by DC as a
single extreme point — remove it and r ≈ 0.11. Rank correlation is ≈ 0.13, so the model
barely predicts the ordering of states.

**The size gradient is largely unreproduced.** In the data, smaller states have much
higher interstate turnover: corr(log population, in-migration rate) = −0.68. The model
reaches −0.22. Two different geometric encodings of state size were tried (see
`compare_assumptions.py`); neither closes the gap, which suggests the real gradient comes
from mechanisms outside the model — students, military postings, transient resource
labour — rather than from geometry.

**No childcare lever.** The proposal has four policy levers, the implementation has
three. Agents carry no family-status attribute, so there is nothing for childcare support
to act on. Adding it means adding a fifth agent attribute.

**The national cohort size is fixed.** The model studies how a constant population of
young adults redistributes, not how many there are. Baseline housing growth is therefore
zero — any positive value makes supply outrun demand everywhere and drives rents down
nationally, swamping the between-state differences of interest.
