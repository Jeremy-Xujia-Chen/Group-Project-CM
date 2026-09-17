# Group-Project-CM

## Project Proposal: A Spatial Agent-Based Model of Young-Adult Interstate Migration and State-Government Policy Competition in the United States

**Working title**

*How Does State Policy Competition Shape Young-Adult Migration and Regional Inequality in the United States? A Spatial Agent-Based Model of Higher Education, Housing, and Labour Markets.*

### Research Objectives

This project aims to build a spatial Agent-Based Model (ABM) to study how U.S. states' education, housing, employment, and childcare policies influence the migration, further-education, and employment decisions of young adults, and, through these individual decisions, further shape the retention of highly educated talent, regional demographic change, and interstate inequality.

The model does not attempt to precisely forecast the future U.S. population. Rather, under explicitly stated assumptions, it compares the long-term effects of different policy combinations.

### Core Research Questions

Under an equal fiscal budget, when college subsidies, housing supply, business/employment incentives, and childcare support are implemented individually or in combination, how do they affect:

- the net interstate migration of young adults aged 18–35;
- the retention rate of highly educated talent;
- each state's employment, income, rent burden, and population age structure;
- interstate policy competition and its spillover effects on neighbouring states?

### Model Scope

- The model selects roughly 8 representative states rather than simulating all 50 U.S. states.
- It focuses on young adults aged 18–35 and young families.
- Each time step represents one year, with a simulation horizon of about 20 years.
- Initial conditions are set using publicly available state-level data, such as the Census ACS, interstate migration flows, BLS employment data, IRS migration-income data, and IPEDS higher-education data.

### Agents and Mechanisms

The model includes four types of agents:

1. **Individual/household agents** — characterised by age, education, skills, income, employment status, family status, student debt, housing burden, and social networks.
2. **State-government agents** — which, under a fixed budget, choose among education, housing, employment, or childcare policies.
3. **Firm agents** — which adjust the number of jobs in response to the supply of high-skilled labour, wages, and market size.
4. **University agents** — which set tuition and enrolment/graduation sizes, thereby influencing the supply of talent within their state.

Individuals decide probabilistically whether to remain in their current state or migrate to another, based on expected real income, job matching, rent, educational opportunities, family ties, migration costs, and policy incentives.

Migration changes the supply of talent, housing demand, rents, firm expansion, and local public finances; these changes in turn affect individuals' decisions in the following year, thus generating dynamic feedback.

### Planned Experiments

- A baseline scenario with no new policies.
- Implementing education subsidies, housing policy, employment incentives, or childcare subsidies individually.
- Comparing different policy combinations under an equal budget.
- Comparing "concentrated investment in a single state" against "evenly distributed investment across states."
- Introducing external shocks such as an economic recession or a rise in remote work.
- Running each scenario across multiple random seeds to test the stability of the results.

### Expected Contributions and Emergent Phenomena

The model focuses on how macro-level outcomes emerge endogenously from dispersed individual decisions, for example:

- the concentration of talent in a few states, or regional brain drain;
- employment growth attracting population, while the resulting housing pressure crowds out lower-income young adults;
- whether education subsidies genuinely retain talent, or merely raise the probability that graduates leave after completing their studies;
- whether one state's policy comes at the cost of talent loss for neighbouring states;
- how small initial differences, through feedback, produce long-term regional divergence and path dependence.

### Feedback Sought

1. Are the choices of 8 states, the 18–35 age range, and a 20-year simulation horizon appropriate in scope?
2. Are the four types of agents overly complex? Should firms or universities be simplified into state-level mechanisms?
3. Are the four policy categories — education, housing, employment, and childcare — sufficient, or should the number of variables be reduced?
4. Is the approach of initialising with publicly available state-level data and aiming at scenario analysis rather than precise prediction consistent with the requirements of the project?
5. Is this topic sufficiently innovative while still being feasible to complete within the timeframe of a course project?
