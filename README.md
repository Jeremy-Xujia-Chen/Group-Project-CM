# Young-Adult Interstate Migration ABM

Code for the CITS4403 Computational Modelling project. By Jeremy Chan (24600092), Jason Tian (23116308) and Joseph Scott (25253165).

A spatial agent-based model of 18–35-year-old migration between eight US states (CA, TX, NY, FL, MA, NC, IL, WA), calibrated to IRS route flows and the ACS PUMS interstate move rate. It compares how strong a targeted housing and employment policy must be to overcome social attachment, and how migration chains and housing capacity amplify or limit the response.

## Repository structure

```
src/                 Model and experiment code
  config.py          Paths, state set, experimental design and seeds
  data.py            Loads the empirical state inputs, distances and IRS routes
  calibration.py     IRS route calibration of destination-choice coefficients
  parameters.py      Model parameters, shock scenarios, policy tables, focal state
  agents.py          Agent initialisation and initial social network
  model.py           PolicySocialABM, the core model
  extensions.py      Distance-decay network and open-system model variants
  experiments.py     Run summaries and per-case experiment runners
utils/               Helper functions
  stats.py           Confidence intervals, power, Hill fits, thresholds
  analysis.py        Cost scaling, net-benefit index, cache verification
  plotting.py        Shared colours and labelling
  parallel.py        Serial / joblib case runner
data/
  raw/               Empirical inputs (state snapshot, IRS routes, ACS PUMS, BRFSS)
  results/           Cached seed-level simulation outputs
  PROVENANCE.csv     Source and run settings for every data file
notebooks/
  project.ipynb      Full analysis: methods, experiments, figures and robustness
requirements.txt     Pinned dependencies
```

## Setup

Python 3.12 is recommended (3.10–3.13 work with the pinned versions).

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Open `notebooks/project.ipynb` in Jupyter or VS Code with the `.venv` interpreter and run all cells (about 30 seconds):

```bash
jupyter notebook notebooks/project.ipynb
```

The notebook loads cached results from `data/results/` so figures and tables reproduce quickly. The final cell (Section 13.6) re-simulates a sample of cases live and checks them against the cache to within 1e-9.

The model can also be used directly from Python, run from the repository root:

```python
from src.experiments import run_lever_case

run_lever_case("NY", "combined", 2.0, "medium", seed=301)
```

## Data

The empirical inputs in `data/raw/` come from the 2023 ACS 1-year estimates and PUMS microdata (U.S. Census Bureau), IRS SOI state-to-state migration data, and CDC BRFSS 2023 module documentation. `data/PROVENANCE.csv` records the source and settings for each file.
