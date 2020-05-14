---
title: "Replication study: Estimating number of infections and impact of NPIs on COVID-19 in European countries (Imperial Report 13)"
author: Tor Erlend Fjelde; Mohamed Tarek; Kai Xu; David Widmann; Martin Trapp; Cameron Pfiffer; Hong Ge 
date: 2020-05-14
---

The Turing.jl team is currently exploring possibilities in an attempt to help with the ongoing SARS-CoV-2 crisis. As preparation for this and to get our feet wet, we decided to perform a replication study of the [Imperial Report 13](https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/covid-19/report-13-europe-npi-impact/), which attempts to estimate the real number of infections and impact of non-pharmaceutical interventions on COVID-19. In the report, the inference was performed using the probabilistic programming language (PPL) Stan. We have explicated their model and inference in Turing.jl, a Julia-based PPL. We believe the results and analysis of our study are relevant for the public, and for other researchers who are actively working on epidemiological models. To that end, our implementation and results are available [here](https://github.com/cambridge-mlg/Covid19).


In summary, we replicated the Imperial COVID-19 model using Turing.jl. Subsequently, we compared the inference results between Turing and Stan, and our comparison indicates that results are reproducible with two different implementations. In particular, we performed 4 sets of simulations using the Imperial COVID-19 model. The resulting estimates of the expected real number of cases, in contrast to the *recorded* number of cases, the reproduction number \\(R\_t\\), and the expected number of deaths as a function of time and non-pharmaceutical interventions (NPIs) for each Simulation are shown below. 




{% include plotly.html id='simulation-1-full' json='../assets/figures/2020-05-04-Imperial-Report13-analysis/full_prior.json' %}

**Simulation (a):** hypothetical Simulation from the model without data (prior predictive) or non-pharmaceutical interventions. Under the prior assumptions of the Imperial Covid-19 model, there is a very wide range of epidemic progressions with expected cases from almost 0 to 100% of the population over time. The black bar corresponds to the date of the last observation. Note that \\(R\_t\\) has a different time-range than the other plots; following the original report, this shows the 100 days following the country-specific `epidemic_start` which is defined to be 31 days prior to the first date of 10 cumulative deaths, while the other plots show the last 60 days.



{% include plotly.html id='simulation-2-full' json='../assets/figures/2020-05-04-Imperial-Report13-analysis/full_posterior.json' %}

**Simulation (b):** future Simulation with non-pharmaceutical interventions kept in place (posterior predictive). After incorporating the observed infection data, we can see a substantially more refined range of epidemic progression. The reproduction rate estimate lies in the range of 3.5-5.6 before any intervention is introduced. The dotted lines correspond to observations, and the black bar corresponds to the date of the last observation.

{% include plotly.html id='simulation-3-full' json='../assets/figures/2020-05-04-Imperial-Report13-analysis/full_counterfactual.json' %}

**Simulation (c):** future Simulation with non-pharmaceutical interventions removed. Now we see the hypothetical scenarios after incorporating infection data, but with non-pharmaceutical interventions removed. This plot looks similar to Simulation (a), but with a more rapid progression of the pandemic since the estimated reproduction rate is bigger than the prior assumptions. The dotted lines correspond to observations, and the black bar corresponds to the date of the last observation.

{% include plotly.html id='simulation-4-full' json='../assets/figures/2020-05-04-Imperial-Report13-analysis/full_counterfactual2.json' %}

**Simulation (d):** future Simulation with when `lockdown` is lifted two weeks before the last observation (predictive posterior). As a result there is a clear, rapid rebound of the reproduction rate. Comparing with Simulation (b) we do not observe an *immediate* increase in the number of expected cases and deaths upon lifting lockdown, but there is a significant difference in the number of cases and deaths in the last few days in the plot: Simulation (d) results in both greater number of cases and deaths, as expected. This demonstrates how the effects of lifting an intervention might not become apparent in the measurable variables, e.g. deaths, until several weeks later. The dotted lines correspond to observations, the black bar corresponds to the date of the last observation, and the red bar indicates when `lockdown` was lifted.

Overall, Simulation (a) shows the prior modelling assumptions, and how these prior assumptions determine the predicted number of cases, etc. before seeing any data. Simulation (b) predicts the trend of the number of cases, etc. using estimated parameters and by keeping all the non-pharmaceutical interventions in place. Simulation (c) shows the estimate in the case where none of the intervention measures are ever put in place. Simulation (d) shows the estimates in the case when the lockdown was lifted two weeks prior to the last observation while keeping all the other non-pharmaceutical interventions in place.

We want to emphasise that we do not provide additional analysis of the Imperial model yet, nor are we aiming to make any claims about the validity or the implications of the model. Instead we refer to Imperial Report 13 for more details and analysis. The purpose of this post is solely to add validation to the *inference* performed in the paper by obtaining the same results using a different probabilistic programming language (PPL) and by exploring whether or not Turing.jl can be useful for researchers working on these problems.

For our next steps, we're looking at collaboration with other researchers and further developments of this and similar models.
There are some immediate directions to explore:
1. Incoporation of more sources of data, e.g. national mobility, seasonal changes and behavior changes in individuals.
2. How the assumptions incorporated into the priors and their parameters change resulting posterior.
3. The current model does not directly include recovery as a possibility and assumes that if a person has been infected once then he/she will be infectious until death. Number of recovered cases suffers from the same issues as the number of cases: it cannot be directly observed. But we can also deal with it in a similar manner as is done with number of cases and incorporate this into the model for a potential improvement.
This will result in a plethora of different models from which we can select the most realistic one using different model comparions techniques, e.g. leave-one-out cross-validation (loo-cv). 

Such model refinement can be potentially valuable given the high impact of this pandemic and the uncertainty and debates in the potential outcomes.

**Acknowledgement** *We would like to thank the Julia community for creating such an excellent platform for scientific computing, and for the continuous feedback that we have received. We also thank researchers from Computational and Biological Laboratory at Cambridge University for their feedback on an early version of the post.*
<!----- Footnotes ----->
