---
title: "Replication study: Estimating number of inections and impact of NPIs on COVID-19 in European countries (Imperial Report 13)"
author: Tor Erlend Fjelde; Mohamed Tarek; Kai Xu; David Widmann; Martin Trapp; Cameron Pfiffer; Hong Ge 
date: 2020-05-04
draft: true
---
<style>div.two-by-two { height: 100%;display: flex;flex-direction: row;flex-wrap: wrap; } div.two-by-two > p { width: 45%; margin: 0 auto; } </style>

We, i.e. the [TuringLang team](https://turing.ml/dev/team/), are currently exploring cooperation with other researchers in attempt to help with the ongoing crisis. As preparation for this and to get our feet wet, we decided it would be useful to do a replication study of the [Imperial Report 13](https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/covid-19/report-13-europe-npi-impact/). We figured it might be useful for the public, in particular other researchers working on the same or similar models, to see the results of this analysis, and thus decided to make it available here.

We want to emphasize that you should look to the original paper rather than this post for developments and analysis of the model. We are not aiming to make any claims about the validity or the implications of the model and refer to [Imperial Report 13](https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/covid-19/report-13-europe-npi-impact/) for details on the model itself. This post's purpose is only to add tiny bit of validation to the *inference* performed in the paper by obtaining the same results using a different probabilistic programming language (PPL) and to explore whether or not `Turing.jl` can be useful for researchers working on these problems.

{% include plotly.html id='plot-3' json='../assets/figures/2020-05-04-Imperial-Report13-analysis/plot2.json' %}

All code and inference results shown in this post can be found [here](https://github.com/TuringLang/Covid19).


# Setup

This is all assuming that you're in the project directory of [`Covid19.jl`](https://github.com/TuringLang/Covid19), a small package where we gather most of our ongoing work.

In the project we use [`DrWatson.jl`](https://github.com/JuliaDynamics/DrWatson.jl) which provides a lot of convenient functionality for, well, working with a project. The below code will activate the `Covid19.jl` project to ensure that we're using correct versions of all the dependencies, i.e. code is reproducible. It's basically just doing `] activate` but with a few bells and whistles that we'll use later.

```julia
using DrWatson
quickactivate(@__DIR__)
```

With the project activate, we can import the `Covid19.jl` package:

```julia
# Loading the project (https://github.com/TuringLang/Covid19)
using Covid19
```

```julia
# Some other packages we'll need
using Random, Dates, Turing, Bijectors
```

And we'll be using the new [multithreading functionality in Julia](https://julialang.org/blog/2019/07/multithreading/) to speed things up, so we need the `Base.Threads` package.

```julia
using Base.Threads
nthreads()
```

    4

Here's a summary of the setup:

```julia
using Pkg
Pkg.status()
```

    Project Covid19 v0.1.0
    Status `~/Projects/mine/Covid19/Project.toml`
      [dce04be8] ArgCheck v2.0.0
      [c7e460c6] ArgParse v1.1.0
      [131c737c] ArviZ v0.4.1
      [76274a88] Bijectors v0.6.7 #tor/using-bijectors-in-link-and-invlink (https://github.com/TuringLang/Bijectors.jl.git)
      [336ed68f] CSV v0.6.1
      [a93c6f00] DataFrames v0.20.2
      [31c24e10] Distributions v0.23.2
      [ced4e74d] DistributionsAD v0.4.10
      [634d3b9d] DrWatson v1.10.2
      [1a297f60] FillArrays v0.8.7
      [58dd65bb] Plotly v0.3.0
      [f0f68f2c] PlotlyJS v0.13.1
      [91a5bcdd] Plots v1.1.2
      [438e738f] PyCall v1.91.4
      [d330b81b] PyPlot v2.9.0
      [df47a6cb] RData v0.7.1
      [2913bbd2] StatsBase v0.33.0
      [f3b207a7] StatsPlots v0.14.5
      [fce5fe82] Turing v0.11.0 #tor/modelling-temporary (https://github.com/TuringLang/Turing.jl.git)
      [9a3f8284] Random 
      [10745b16] Statistics 

In the Github project you will find a `Manifest.toml`. This means that if you're working directory is the project directory, and you do `julia --project` followed by `] instantiate` you will have *exactly* the same enviroment as we had when performing this analysis.

{% details Overloading some functionality in `DrWatson.jl` %}

As mentioned `DrWatson.jl` provides a lot of convenience functions for working in a project, and one of them is `projectdir(args...)` which will resolve join the `args` to the absolute path of the project directory. In similar fashion it provides a `datadir` method which defaults to `projectdir("data")`. But to remove the possibility of making a type later on when writing `datadir("imperial-report13")`, we'll overload this method for this notebook:

```julia
import DrWatson: datadir

outdir() = projectdir("out")
outdir(args...) = projectdir("out", args...)

datadir() = projectdir("data", "imperial-report13")
datadir(s...) = projectdir("data", "imperial-report13", s...)
```

    datadir (generic function with 2 methods)

{% enddetails %}


# Data

To ensure consistency with the original model from the paper (and to stay up-to-date with the changes made), we preprocessed the data using the `base.r` script from the [original repository (#6ee3010)](https://github.com/ImperialCollegeLondon/covid19model/tree/6ee3010a58a57cc14a16545ae897ca668b7c9096) and store the processed data in a `processed.rds` file. To load this RDS file obtained from the R script, we make use of the RData.jl package which allows us to load the RDS file into a Julia `Dict`.

```julia
using RData
```

```julia
rdata_full = load(datadir("processed.rds"))
rdata = rdata_full["stan_data"];
```

```julia
keys(rdata_full)
```

    Base.KeySet for a Dict{String,Int64} with 4 entries. Keys:
      "reported_cases"
      "stan_data"
      "deaths_by_country"
      "dates"

```julia
country_to_dates = d = Dict([(k, rdata_full["dates"][k]) for k in keys(rdata_full["dates"])])
```

    Dict{String,Array{Date,1}} with 14 entries:
      "Sweden"         => Date[2020-02-18, 2020-02-19, 2020-02-20, 2020-02-21, 2020…
      "Belgium"        => Date[2020-02-18, 2020-02-19, 2020-02-20, 2020-02-21, 2020…
      "Greece"         => Date[2020-02-19, 2020-02-20, 2020-02-21, 2020-02-22, 2020…
      "Switzerland"    => Date[2020-02-14, 2020-02-15, 2020-02-16, 2020-02-17, 2020…
      "Germany"        => Date[2020-02-15, 2020-02-16, 2020-02-17, 2020-02-18, 2020…
      "United_Kingdom" => Date[2020-02-12, 2020-02-13, 2020-02-14, 2020-02-15, 2020…
      "Denmark"        => Date[2020-02-21, 2020-02-22, 2020-02-23, 2020-02-24, 2020…
      "Norway"         => Date[2020-02-24, 2020-02-25, 2020-02-26, 2020-02-27, 2020…
      "France"         => Date[2020-02-07, 2020-02-08, 2020-02-09, 2020-02-10, 2020…
      "Portugal"       => Date[2020-02-21, 2020-02-22, 2020-02-23, 2020-02-24, 2020…
      "Spain"          => Date[2020-02-09, 2020-02-10, 2020-02-11, 2020-02-12, 2020…
      "Netherlands"    => Date[2020-02-14, 2020-02-15, 2020-02-16, 2020-02-17, 2020…
      "Italy"          => Date[2020-01-27, 2020-01-28, 2020-01-29, 2020-01-30, 2020…
      "Austria"        => Date[2020-02-22, 2020-02-23, 2020-02-24, 2020-02-25, 2020…

Since the data-format is not native Julia there might be some discrepancies in the *types* of the some data fields, and so we need to do some type-conversion of the loaded `rdata`. We also rename a lot of the fields to make it more understandable and for an easier mapping to our implementation of the model. Feel free to skip this snippet.

{% details Data wrangling %}

```julia
# Convert some misparsed fields
rdata["N2"] = Int(rdata["N2"]);
rdata["N0"] = Int(rdata["N0"]);

rdata["EpidemicStart"] = Int.(rdata["EpidemicStart"]);

rdata["cases"] = Int.(rdata["cases"]);
rdata["deaths"] = Int.(rdata["deaths"]);

# Stan will fail if these are `nothing` so we make them empty arrays
rdata["x"] = []
rdata["features"] = []

countries = (
  "Denmark",
  "Italy",
  "Germany",
  "Spain",
  "United_Kingdom",
  "France",
  "Norway",
  "Belgium",
  "Austria", 
  "Sweden",
  "Switzerland",
  "Greece",
  "Portugal",
  "Netherlands"
)
num_countries = length(countries)

names_covariates = ("schools_universities", "self_isolating_if_ill", "public_events", "any", "lockdown", "social_distancing_encouraged")
lockdown_index = findfirst(==("lockdown"), names_covariates)


function rename!(d, names::Pair...)
    # check that keys are not yet present before updating `d`
    for k_new in values.(names)
        @assert k_new ∉ keys(d) "$(k_new) already in dictionary"
    end

    for (k_old, k_new) in names
        d[k_new] = pop!(d, k_old)
    end
    return d
end

# `rdata` is a `DictOfVector` so we convert to a simple `Dict` for simplicity
d = Dict([(k, rdata[k]) for k in keys(rdata)]) # `values(df)` and `keys(df)` have different ordering so DON'T do `Dict(keys(df), values(df))`

# Rename some columns
rename!(
    d,
    "f" => "π", "SI" => "serial_intervals", "pop" => "population",
    "M" => "num_countries", "N0" => "num_impute", "N" => "num_obs_countries",
    "N2" => "num_total_days", "EpidemicStart" => "epidemic_start",
    "X" => "covariates", "P" => "num_covariates"
)

# Add some type-information to arrays and replace `-1` with `missing` (as `-1` is supposed to represent, well, missing data)
d["deaths"] = Int.(d["deaths"])
# d["deaths"] = replace(d["deaths"], -1 => missing)
d["deaths"] = collect(eachcol(d["deaths"])) # convert into Array of arrays instead of matrix

d["cases"] = Int.(d["cases"])
# d["cases"] = replace(d["cases"], -1 => missing)
d["cases"] = collect(eachcol(d["cases"])) # convert into Array of arrays instead of matrix

d["num_covariates"] = Int(d["num_covariates"])
d["num_countries"] = Int(d["num_countries"])
d["num_total_days"] = Int(d["num_total_days"])
d["num_impute"] = Int(d["num_impute"])
d["num_obs_countries"] = Int.(d["num_obs_countries"])
d["epidemic_start"] = Int.(d["epidemic_start"])
d["population"] = Int.(d["population"])

d["π"] = collect(eachcol(d["π"])) # convert into Array of arrays instead of matrix

# Convert 3D array into Array{Matrix}
covariates = [rdata["X"][m, :, :] for m = 1:num_countries]

data = (; (k => d[String(k)] for k in [:num_countries, :num_impute, :num_obs_countries, :num_total_days, :cases, :deaths, :π, :epidemic_start, :population, :serial_intervals])...)
data = merge(data, (covariates = covariates, ));

# Can deal with ragged arrays, so we can shave off unobserved data (future) which are just filled with -1
data = merge(
    data,
    (cases = [data.cases[m][1:data.num_obs_countries[m]] for m = 1:data.num_countries],
     deaths = [data.deaths[m][1:data.num_obs_countries[m]] for m = 1:data.num_countries])
);
```

{% enddetails %}

```julia
data.num_countries
```

    14

Because it's a bit much to visualize 14 countries at each step, we're going to use UK as an example throughout.

```julia
uk_index = findfirst(==("United_Kingdom"), countries)
```

    5

It's worth noting that the data user here is not quite up-to-date for UK because on <span class="timestamp-wrapper"><span class="timestamp">&lt;2020-04-30 to.&gt; </span></span> they updated their *past* numbers by including deaths from care- and nursing-homes (data source: [ECDC](https://www.ecdc.europa.eu/en)). Thus if you compare the prediction of the model to real numbers, it's likely that the real numbers will be a bit higher than what the model predicts.


# Model

For a thorough description of the model and the assumptions that have gone into it, we recommend looking at the [original paper](https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/covid-19/report-13-europe-npi-impact/) or their very nice [techical report from the repository](https://github.com/ImperialCollegeLondon/covid19model/tree/6ee3010a58a57cc14a16545ae897ca668b7c9096/Technical_description_of_Imperial_COVID_19_Model.pdf). The model described here is the one corresponding to the technical report linked. The link points to the correct commit ID and so should be consistent with this post despite potential changes to the "official" model made in the future.

For the sake of exposition, we present a compact version of the model here:

\begin{align}
  \tau & \sim \mathrm{Exponential}(1 / 0.03) \\\\\\
  y\_m & \sim \mathrm{Exponential}(\tau) \quad & \text{for} \quad m = 1, \dots, M \\\\\\
  \kappa & \sim \mathcal{N}^{ + }(0, 0.5) \\\\\\
  \mu\_m & \sim \mathcal{N}^{ + }(3.28, \kappa) \quad & \text{for} \quad m = 1, \dots, M \\\\\\
  \gamma & \sim \mathcal{N}^{ + }(0, 0.2) \\\\\\
  \beta\_m & \sim \mathcal{N}(0, \gamma) \quad & \text{for} \quad m = 1, \dots, M \\\\\\
  \tilde{\alpha}\_k &\sim \mathrm{Gamma}(0.1667, 1) \quad & \text{for} \quad k = 1, \dots, K \\\\\\
  \alpha\_k &= \tilde{\alpha}\_k - \frac{\log(1.05)}{6} \quad & \text{for} \quad  k = 1, \dots, K \\\\\\
  R\_{t, m} &= \mu\_m \exp(- \beta\_m x\_{k\_{\text{ld}}} - \sum\_{k=1}^{K} \alpha\_k x\_k) \quad & \text{for} \quad m = 1, \dots, M, \ t = 1, \dots, T  \\\\\\
  \tilde{R}\_{t, m} &= 1 - \frac{1}{p\_m} \sum\_{\tau = 1}^{t - 1} c\_{\tau, m}  \quad & \text{for} \quad m = 1, \dots, M, \ t = T\_{\text{impute}} + 1, \dots, T \\\\\\
  c\_{t, m} &= y\_m \quad & \text{for} \quad m = 1, \dots, M, \ t = 1, \dots, T\_{\text{impute}} \\\\\\
  c\_{t, m} &= \tilde{R}\_{t, m} \sum\_{\tau = 1}^{t - 1} c\_{\tau, m} s\_{t - \tau} \quad & \text{for} \quad m = 1, \dots, M, \ t = T\_{\text{impute}} + 1, \dots, T \\\\\\
  \varepsilon\_m^{\text{ifr}} &\sim \mathcal{N}(1, 0.1)^{ + } \quad & \text{for} \quad m = 1, \dots, M \\\\\\
  \mathrm{ifr}\_m^{ \* } &\sim \mathrm{ifr}\_m \cdot \varepsilon\_m^{\text{ifr}} \quad & \text{for} \quad m = 1, \dots, M \\\\\\
  d\_{t, m} &= \mathrm{ifr}\_m^{ \* } \sum\_{\tau=1}^{t - 1} c\_{\tau, m} \pi\_{t - \tau} \quad & \text{for} \quad m = 1, \dots, M, \ t = 1, \dots, T \\\\\\
  \phi  & \sim \mathcal{N}^{ + }(0, 5) \\\\\\
  D\_{t, m} &\sim \mathrm{NegativeBinomial}(d\_{t, m}, \phi) \quad & \text{for} \quad m = 1, \dots, M, \ t = 1, \dots, T 
\end{align}

where

-   it's assumed that seeding of new infections begins 30 days before the day after a country has cumulative observed 10 deaths
-   \\(M\\) denotes the number of countries
-   \\(T\\) the total number of time-steps
-   \\(T\_{\text{impute}}\\) the time steps to *impute* values for; the first 6 of the 30 days we impute the number, and then we simulate the rest
-   \\(\alpha\_k\\) denotes the weights for the k-th intervention/covariate
-   \\(\beta\_m\\) denotes the weight for the `lockdown` intervention (whose index we denote by \\(k\_{\text{ld}}\\))
    -   Note that there is also a \\(\alpha\_{k\_{\text{ld}}}\\) which is shared between all the \\(M\\) countries
    -   In contrast, the \\(\beta\_m\\) weight is local to the country with index \\(m\\)
    -   This is a sort of  way to try and deal with the fact that `lockdown` means different things in different countries, e.g. `lockdown` in UK is much more severe than "lockdown" in Norway.
-   \\(\mu\_m\\) represents the \\(R\_0\\) value for country \\(m\\) (i.e. \\(R\_t\\) without any interventions)
-   \\(R\_{t, m}\\) denotes the **reproduction number** at time \\(t\\) for country \\(m\\)
-   \\(\tilde{R}\_{t, m}\\) denotes the **adjusted reproduction number** at time \\(t\\) for country \\(m\\), *adjusted* in the sense that it's rescaled wrt. what proportion of the population is susceptible for infection (assuming infected people cannot get the virus again within the near future)
-   \\(p\_{m}\\) denotes the **total/initial population** for country \\(m\\)
-   \\(\mathrm{ifr}\_m\\) denotes the **infection-fatality ratio** for country \\(m\\), and \\(\mathrm{ifr}\_m^{ \* }\\) the *adjusted* infection-fatality ratio (see paper)
-   \\(\varepsilon\_m^{\text{ifr}}\\) denotes the noise for the multiplicative noise for the \\(\mathrm{ifr}\_m^{ \* }\\)
-   \\(\pi\\) denotes the **time from infection to death** and is assumed to be a sum of two independent random times: the incubation period (*infection-to-onset*) and time between onset of symptoms and death (*onset-to-death*):
    
    \begin{equation\*}
    \pi \sim \mathrm{Gamma}(5.1, 0.86) + \mathrm{Gamma}(18.8, 0.45)
    \end{equation\*}
    
    where in this case the \\(\mathrm{Gamma}\\) is parameterized by its mean and coefficient of variation. In the model, this is a *precomputed* quantity and not something to be inferred, though in an ideal world this would also be included in the model.
-   \\(\pi\_t\\) then denotes a discretized version of the PDF for \\(\pi\\). The reasoning behind the discretization is that if we assume \\(d\_m(t)\\) to be a continuous random variable denoting the death-rate at any time \\(t\\), then it would be given by
    
    \begin{equation\*}
    d\_m(t) = \mathrm{ifr}\_m^{ \* } \int\_0^t c\_m(\tau) \pi(t - \tau) dt
    \end{equation\*}
    
    i.e. the convolution of the number of cases observed at time time \\(\tau\\), \\(c\_m(\tau)\\), and the *probability* of death at prior to time \\(t\\) for the new cases observed at time \\(\tau\\), \\(\pi(t - \tau)\\) (assuming stationarity of \\(\pi(t)\\)). Thus, \\(c\_m(\tau) \pi(t - \tau)\\) can be interpreted as the portion people who got the virus at time \\(\tau\\) have died at time \\(t\\) (or rather, have died after having the virus for \\(t - \tau\\) time, with \\(t > \tau\\)). Discretizing then results in the above model.
-   \\(s\_t\\) denotes the **serial intervals**, i.e. the time between successive cases in a chain of transmission, also a precomputed quantity
-   \\(c\_{t, m}\\) denotes the **expected daily cases** at time \\(t\\) for country \\(m\\)
-   \\(C\_{t, m}\\) denotes the **cumulative cases** prior to time \\(t\\) for country \\(m\\)
-   \\(d\_{t, m}\\) denotes the **expected daily deaths** at time \\(t\\) for country \\(m\\)
-   \\(D\_{t, m}\\) denotes the **daily deaths** at time \\(t\\) for country \\(m\\) (in our case, this is the **likelihood**); note that here we're using the mean and variance coefficient parameterization of \\(\mathrm{NegativeBinomial}\\)

To see the reasoning for the choices of distributions and parameters for the priors, see the either the paper or the [techical report from the repository](https://github.com/ImperialCollegeLondon/covid19model/tree/6ee3010a58a57cc14a16545ae897ca668b7c9096/Technical_description_of_Imperial_COVID_19_Model.pdf).


## Code

In `Turing.jl`, a "sample"-statement is defined by `x ~ Distribution`. Therefore, the priors used in the model can be written within a Turing.jl `Model` as:

```julia
τ ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
y ~ filldist(Exponential(τ), num_countries)
ϕ ~ truncated(Normal(0, 5), 0, Inf)
κ ~ truncated(Normal(0, 0.5), 0, Inf)
μ ~ filldist(truncated(Normal(3.28, κ), 0, Inf), num_countries)

α_hier ~ filldist(Gamma(.1667, 1), num_covariates)
α = α_hier .- log(1.05) / 6.

ifr_noise ~ filldist(truncated(Normal(1., 0.1), 0, Inf), num_countries)

# lockdown-related
γ ~ truncated(Normal(0, 0.2), 0, Inf)
lockdown ~ filldist(Normal(0, γ), num_countries)
```

The `filldist` function in the above snippet is a function used to construct a `Distribution` from which we can obtain i.i.d. samples from a univariate distribution using vectorization.

And the full model is defined:

```julia
@model function model_v2(
    num_impute,        # [Int] num. of days for which to impute infections
    num_total_days,    # [Int] days of observed data + num. of days to forecast
    cases,             # [AbstractVector{<:AbstractVector{<:Int}}] reported cases
    deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
    π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    covariates,        # [Vector{<:AbstractMatrix}]
    epidemic_start,    # [AbstractVector{<:Int}]
    population,        # [AbstractVector{<:Real}]
    serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
    lockdown_index,    # [Int] the index for the `lockdown` covariate in `covariates`
    predict=false,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    ::Type{TV} = Vector{Float64}
) where {TV}
    # `covariates` should be of length `num_countries` and each entry correspond to a matrix of size `(num_total_days, num_covariates)`
    num_covariates = size(covariates[1], 2)
    num_countries = length(cases)
    num_obs_countries = length.(cases)

    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    last_time_steps = predict ? fill(num_total_days, num_countries) : num_obs_countries

    # Latent variables
    τ ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
    y ~ filldist(Exponential(τ), num_countries)
    ϕ ~ truncated(Normal(0, 5), 0, Inf)
    κ ~ truncated(Normal(0, 0.5), 0, Inf)
    μ ~ filldist(truncated(Normal(3.28, κ), 0, Inf), num_countries)

    α_hier ~ filldist(Gamma(.1667, 1), num_covariates)
    α = α_hier .- log(1.05) / 6.

    ifr_noise ~ filldist(truncated(Normal(1., 0.1), 0, Inf), num_countries)

    # lockdown-related
    γ ~ truncated(Normal(0, 0.2), 0, Inf)
    lockdown ~ filldist(Normal(0, γ), num_countries)

    # Initialization of some quantities
    expected_daily_cases = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    cases_pred = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    expected_daily_deaths = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    Rt = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    Rt_adj = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]

    # Loops over countries and perform independent computations for each country
    # since this model does not include any notion of migration across borders.
    # => might has well wrap it in a `@threads` to perform the computation in parallel.
    @threads for m = 1:num_countries
        # Country-specific parameters
        π_m = π[m]
        pop_m = population[m]
        expected_daily_cases_m = expected_daily_cases[m]
        cases_pred_m = cases_pred[m]
        expected_daily_deaths_m = expected_daily_deaths[m]
        Rt_m = Rt[m]
        Rt_adj_m = Rt_adj[m]

        last_time_step = last_time_steps[m]

        # Imputation of `num_impute` days
        expected_daily_cases_m[1:num_impute] .= y[m]
        cases_pred_m[1] = zero(cases_pred_m[1])
        cases_pred_m[2:num_impute] .= cumsum(expected_daily_cases_m[1:num_impute - 1])

        xs = covariates[m][1:last_time_step, :] # extract covariates for the wanted time-steps and country `m`
        Rt_m .= μ[m] * exp.(xs * (-α) + (- lockdown[m]) * xs[:, lockdown_index])

        # Adjusts for portion of pop that are susceptible
        Rt_adj_m[1:num_impute] .= (max.(pop_m .- cases_pred_m[1:num_impute], zero(cases_pred_m[1])) ./ pop_m) .* Rt_m[1:num_impute]

        for t = (num_impute + 1):last_time_step
            # Update cumulative cases
            cases_pred_m[t] = cases_pred_m[t - 1] + expected_daily_cases_m[t - 1]

            # Adjusts for portion of pop that are susceptible
            Rt_adj_m[t] = (max(pop_m - cases_pred_m[t], zero(cases_pred_m[t])) / pop_m) * Rt_m[t]

            expected_daily_cases_m[t] = Rt_adj_m[t] * sum(expected_daily_cases_m[τ] * serial_intervals[t - τ] for τ = 1:(t - 1))
        end

        expected_daily_deaths_m[1] = 1e-15 * expected_daily_cases_m[1]
        for t = 2:last_time_step
            expected_daily_deaths_m[t] = sum(expected_daily_cases_m[τ] * π_m[t - τ] * ifr_noise[m] for τ = 1:(t - 1))
        end
    end

    # Observe
    for m = 1:num_countries
        # Extract the estimated expected daily deaths for country `m`
        expected_daily_deaths_m = expected_daily_deaths[m]
        # Extract time-steps for which we have observations
        ts = epidemic_start[m]:num_obs_countries[m]
        # Observe!
        deaths[m][ts] ~ arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ))
    end

    return (
        expected_daily_cases = expected_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        Rt = Rt,
        Rt_adjusted = Rt_adj
    )
end;
```

Two things worth noting is the use of the `TV` variable to instantiate some internal variables and the use of `@threads`. 

As you can see in the arguments for the model, `TV` refers to a *type* and will be recognized as such by the `@model` macro when transforming the model code. This is used to ensure *type-stability* of the model.

{% details More detailed explanation of `TV` %}

A default execution of the model will then use `TV` as `Vector{Float64}`, thus making statements like `TV(undef, n)` result in a `Vector{Float64}` with `undef` (uninitialized) values and length `n`. But in case we use a sampler that requires automatic differentiation (AD), TV will be will be replaced with the AD-type corresponding to a `Vector`, e.g. `TrackedVector` in the case of [`Tracker.jl`](https://github.com/FluxML/Tracker.jl).

{% enddetails %}

The use of `@threads` means that inside each execution of the `Model`, this loop will be performed in parallel, where the number of threads are specified by the enviroment variable `JULIA_NUM_THREADS`. This is thanks to the really nice multithreading functionality [introduced in Julia 1.3](https://julialang.org/blog/2019/07/multithreading/) (and so this also requires Julia 1.3 or higher to run the code). Note that the inside the loop is independent of each other and each `m` will be seen by only one thread, hence it's threadsafe.

This model is basically identitical to the one defined in [stan-models/base.stan (#6ee3010)](https://github.com/ImperialCollegeLondon/covid19model/blob/6ee3010a58a57cc14a16545ae897ca668b7c9096/stan-models/base.stan) with the exception of two points:

-   In this model we use `TruncatedNormal` for normally distributed variables which are positively constrained instead of sampling from a `Normal` and then taking the absolute value; these approaches are equivalent from a modelling perspective.
-   We've added the use of `max(pop_m - cases_pred_m[t], 0)` in computing the *adjusted* \\(R\_t\\), `Rt_adj`, to ensure that in the case where the entire populations has died there, the adjusted \\(R\_t\\) is set to 0, i.e. if everyone in the country passed away then there is no spread of the virus (this does not affect "correctness" of inference). [^fn1]
-   The `cases` and `deaths` arguments are arrays of arrays instead of 3D arrays, therefore we don't need to fill the future days with `-1` as is done in the original model.


### Multithreaded observe

We can also make the `observe` statements parallel, but because the `~` is not (yet) threadsafe we unfortunately have to touch some of the internals of `Turing.jl`. But for observations it's very straight-forward: instead of observing by the following piece of code

```julia
for m = 1:num_countries
    # Extract the estimated expected daily deaths for country `m`
    expected_daily_deaths_m = expected_daily_deaths[m]
    # Extract time-steps for which we have observations
    ts = epidemic_start[m]:num_obs_countries[m]
    # Observe!
    deaths[m][ts] ~ arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ))
end
```

we can use the following

```julia
# Doing observations in parallel provides a small speedup
logps = TV(undef, num_countries)
@threads for m = 1:num_countries
    # Extract the estimated expected daily deaths for country `m`
    expected_daily_deaths_m = expected_daily_deaths[m]
    # Extract time-steps for which we have observations
    ts = epidemic_start[m]:num_obs_countries[m]
    # Observe!
    logps[m] = logpdf(arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ)), deaths[m][ts])
end
Turing.acclogp!(_varinfo, sum(logps))
```

{% details Explanation of what we just did %}

It might be worth explaining a bit about what's going on here. First we should explain what the deal is with `_varinfo`. `_varinfo` is basically the object used internally in Turing to track the sampled variables and the log-pdf *for a particular evaluation* of the model, and so `acclogp!(_varinfo, lp)` will increment the log-pdf stored in `_varinfo` by `lp`. With that we can explain what happens to `~` inside the `@macro`. Using the old observe-snippet as an example, the `@model` macro replaces `~` with

```julia
acclogp!(_varinfo., logpdf(arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ)), deaths[m][ts]))
```

But we're iterating through `m`, so this would not be thread-safe since you might be two threads attempting to mutate `_varinfo` simultaneously.[^fn2] Therefore, since no threads sees the same `m`, delaying the accumulation to after having computed all the log-pdf in parallel leaves us with equivalent code that is threadsafe.

You can read more about the `@macro` and its internals [here](https://turing.ml/dev/docs/for-developers/compiler#model-macro-and-modelgen).

{% enddetails %}


### Final model

This results in the following model definition

```julia
@model function model_v2(
    num_impute,        # [Int] num. of days for which to impute infections
    num_total_days,    # [Int] days of observed data + num. of days to forecast
    cases,             # [AbstractVector{<:AbstractVector{<:Int}}] reported cases
    deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
    π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    covariates,        # [Vector{<:AbstractMatrix}]
    epidemic_start,    # [AbstractVector{<:Int}]
    population,        # [AbstractVector{<:Real}]
    serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
    lockdown_index,    # [Int] the index for the `lockdown` covariate in `covariates`
    predict=false,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    ::Type{TV} = Vector{Float64}
) where {TV}
    # `covariates` should be of length `num_countries` and each entry correspond to a matrix of size `(num_total_days, num_covariates)`
    num_covariates = size(covariates[1], 2)
    num_countries = length(cases)
    num_obs_countries = length.(cases)

    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    last_time_steps = predict ? fill(num_total_days, num_countries) : num_obs_countries

    # Latent variables
    τ ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
    y ~ filldist(Exponential(τ), num_countries)
    ϕ ~ truncated(Normal(0, 5), 0, Inf)
    κ ~ truncated(Normal(0, 0.5), 0, Inf)
    μ ~ filldist(truncated(Normal(3.28, κ), 0, Inf), num_countries)

    α_hier ~ filldist(Gamma(.1667, 1), num_covariates)
    α = α_hier .- log(1.05) / 6.

    ifr_noise ~ filldist(truncated(Normal(1., 0.1), 0, Inf), num_countries)

    # lockdown-related
    γ ~ truncated(Normal(0, 0.2), 0, Inf)
    lockdown ~ filldist(Normal(0, γ), num_countries)

    # Initialization of some quantities
    expected_daily_cases = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    cases_pred = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    expected_daily_deaths = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    Rt = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    Rt_adj = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]

    # Loops over countries and perform independent computations for each country
    # since this model does not include any notion of migration across borders.
    # => might has well wrap it in a `@threads` to perform the computation in parallel.
    @threads for m = 1:num_countries
        # Country-specific parameters
        π_m = π[m]
        pop_m = population[m]
        expected_daily_cases_m = expected_daily_cases[m]
        cases_pred_m = cases_pred[m]
        expected_daily_deaths_m = expected_daily_deaths[m]
        Rt_m = Rt[m]
        Rt_adj_m = Rt_adj[m]

        last_time_step = last_time_steps[m]

        # Imputation of `num_impute` days
        expected_daily_cases_m[1:num_impute] .= y[m]
        cases_pred_m[1] = zero(cases_pred_m[1])
        cases_pred_m[2:num_impute] .= cumsum(expected_daily_cases_m[1:num_impute - 1])

        xs = covariates[m][1:last_time_step, :] # extract covariates for the wanted time-steps and country `m`
        Rt_m .= μ[m] * exp.(xs * (-α) + (- lockdown[m]) * xs[:, lockdown_index])

        # Adjusts for portion of pop that are susceptible
        Rt_adj_m[1:num_impute] .= (max.(pop_m .- cases_pred_m[1:num_impute], zero(cases_pred_m[1])) ./ pop_m) .* Rt_m[1:num_impute]

        for t = (num_impute + 1):last_time_step
            # Update cumulative cases
            cases_pred_m[t] = cases_pred_m[t - 1] + expected_daily_cases_m[t - 1]

            # Adjusts for portion of pop that are susceptible
            Rt_adj_m[t] = (max(pop_m - cases_pred_m[t], zero(cases_pred_m[t])) / pop_m) * Rt_m[t]

            expected_daily_cases_m[t] = Rt_adj_m[t] * sum(expected_daily_cases_m[τ] * serial_intervals[t - τ] for τ = 1:(t - 1))
        end

        expected_daily_deaths_m[1] = 1e-15 * expected_daily_cases_m[1]
        for t = 2:last_time_step
            expected_daily_deaths_m[t] = sum(expected_daily_cases_m[τ] * π_m[t - τ] * ifr_noise[m] for τ = 1:(t - 1))
        end
    end

    # Observe
    # Doing observations in parallel provides a small speedup
    logps = TV(undef, num_countries)
    @threads for m = 1:num_countries
        # Extract the estimated expected daily deaths for country `m`
        expected_daily_deaths_m = expected_daily_deaths[m]
        # Extract time-steps for which we have observations
        ts = epidemic_start[m]:num_obs_countries[m]
        # Observe!
        logps[m] = logpdf(arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ)), deaths[m][ts])
    end
    Turing.acclogp!(_varinfo, sum(logps))

    return (
        expected_daily_cases = expected_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        Rt = Rt,
        Rt_adjusted = Rt_adj
    )
end;
```

    ┌ Warning: you are using the internal variable `_varinfo`
    └ @ DynamicPPL /homes/tef30/.julia/packages/DynamicPPL/3jy49/src/compiler.jl:175

We define an alias `model_def` so that if we want to try out a different model, there's only one point in the notebook which we need to change.

```julia
model_def = model_v2;
```

The input data have up to 30-40 days of unobserved future data which we might want to predict on. But during sampling we don't want to waste computation on sampling for the future for which we do not have any observations. Therefore we have an argument `predict::Bool` in the model which allows us to specify whether or not to generate future quantities.

```julia
# Model instantance used to for inference
m_no_pred = model_def(
    data.num_impute,
    data.num_total_days,
    data.cases,
    data.deaths,
    data.π,
    data.covariates,
    data.epidemic_start,
    data.population,
    data.serial_intervals,
    lockdown_index,
    false # <= DON'T predict
);
```

```julia
# Model instance used for prediction
m = model_def(
    data.num_impute,
    data.num_total_days,
    data.cases,
    data.deaths,
    data.π,
    data.covariates,
    data.epidemic_start,
    data.population,
    data.serial_intervals,
    lockdown_index,
    true # <= predict
);
```

Just to make sure everything is working, we can "evaluate" the model to obtain a sample from the prior:

```julia
res = m();
res.expected_daily_cases[uk_index]
```

    100-element Array{Float64,1}:
     0.2978422827487166
     0.2978422827487166
     0.2978422827487166
     0.2978422827487166
     0.2978422827487166
     0.2978422827487166
     0.832817101553483
     1.0346063917235566
     1.3678178161493468
     1.8604943842674377
     2.546129501657631
     3.4852378065416523
     4.768875412263893
     ⋮
     0.24406116233616038
     0.17186744515247868
     0.1208474970159245
     0.08485139699103932
     0.05949579772505278
     0.04166274540068994
     0.029138775736189983
     0.02035555519759347
     0.014203911930339133
     0.009900796534873262
     0.00689432670057081
     0.004796166095549548


# Visualization utilities

For visualisation we of course use [Plots.jl](https://github.com/JuliaPlots/Plots.jl), and in this case we're going to use the `pyplot` backend which uses Python's matplotlib under the hood.

```julia
using Plots, StatsPlots
```

{% details Method definition for plotting the predictive distribution %}

```julia
# Ehh, this can be made nicer...
function country_prediction_plot(country_idx, predictions_country::AbstractMatrix, e_deaths_country::AbstractMatrix, Rt_country::AbstractMatrix; normalize_pop::Bool = false, main_title="")
    pop = data.population[country_idx]
    num_total_days = data.num_total_days
    num_observed_days = length(data.cases[country_idx])

    country_name = countries[country_idx]
    start_date = first(country_to_dates[country_name])
    dates = cumsum(fill(Day(1), data.num_total_days)) + (start_date - Day(1))
    date_strings = Dates.format.(dates, "Y-mm-dd")

    # A tiny bit of preprocessing of the data
    preproc(x) = normalize_pop ? x ./ pop : x

    daily_deaths = data.deaths[country_idx]
    daily_cases = data.cases[country_idx]

    p1 = plot(; xaxis = false, legend = :topleft)
    bar!(preproc(daily_deaths), label="Observed daily deaths")
    title!(replace(country_name, "_" => " ") * " " * main_title)
    vline!([data.epidemic_start[country_idx]], label="epidemic start", linewidth=2)
    vline!([num_observed_days], label="end of observations", linewidth=2)
    xlims!(0, num_total_days)

    p2 = plot(; legend = :topleft, xaxis=false)
    plot_confidence_timeseries!(p2, preproc(e_deaths_country); label = "Expected daily deaths")
    # title!("Expected daily deaths (pred)")
    bar!(preproc(daily_deaths), label="Recorded daily deaths (observed)", alpha=0.5)

    p3 = plot(; legend = :bottomleft, xaxis=false)
    plot_confidence_timeseries!(p3, Rt_country; no_label = true)
    for (c_idx, c_time) in enumerate(findfirst.(==(1), eachcol(data.covariates[country_idx])))
        if c_time !== nothing
            c_name = names_covariates[c_idx]
            if (c_name != "any")
                # Don't add the "any intervention" stuff
                vline!([c_time - 1], label=c_name)
            end
        end
    end
    title!("Rt")
    qs = [quantile(v, [0.025, 0.975]) for v in eachrow(Rt_country)]
    lq, hq = (eachrow(hcat(qs...))..., )
    ylims!(0, maximum(hq) + 0.1)

    p4 = plot(; legend = :topleft, xaxis=false)
    plot_confidence_timeseries!(p4, preproc(predictions_country); label = "Expected daily cases")
    # title!("Expected daily cases (pred)")
    bar!(preproc(daily_cases), label="Recorded daily cases (observed)", alpha=0.5)

    vals = preproc(cumsum(e_deaths_country; dims = 1))
    p5 = plot(; legend = :topleft, xaxis=false)
    plot_confidence_timeseries!(p5, vals; label = "Expected deaths")
    plot!(preproc(cumsum(daily_deaths)), label="Recorded deaths (observed)", color=:red)
    # title!("Expected deaths (pred)")

    vals = preproc(cumsum(predictions_country; dims = 1))
    p6 = plot(; legend = :topleft)
    plot_confidence_timeseries!(p6, vals; label = "Expected cases")
    plot!(preproc(daily_cases), label="Recorded cases (observed)", color=:red)
    # title!("Expected cases (pred)")

    p = plot(p1, p3, p2, p4, p5, p6, layout=(6, 1), size=(900, 1200), sharex=true)
    xticks!(1:3:num_total_days, date_strings[1:3:end], xrotation=45)

    return p
end

function country_prediction_plot(country_idx, cases, e_deaths, Rt; kwargs...)
    n = length(cases)
    e_deaths_country = hcat([e_deaths[t][country_idx] for t = 1:n]...)
    Rt_country = hcat([Rt[t][country_idx] for t = 1:n]...)
    predictions_country = hcat([cases[t][country_idx] for t = 1:n]...)

    return country_prediction_plot(country_idx, predictions_country, e_deaths_country, Rt_country; kwargs...)
end
```

    country_prediction_plot (generic function with 2 methods)

```julia
function arrarrarr2arr(a::AbstractVector{<:AbstractVector{<:AbstractVector{T}}}) where {T<:Real}
    n1, n2, n3 = length(a), length(first(a)), length(first(first(a)))

    A = zeros(T, (n1, n2, n3))
    for i = 1:n1
        for j = 1:n2
            for k = 1:n3
                A[i, j, k] = a[i][j][k]
            end
        end
    end

    return A
end

function country_cumulative_prediction(vals::AbstractArray{<:Real, 3}; normalize_pop = false, no_label=false, kwargs...)
    lqs, mqs, hqs = [], [], []
    labels = []

    for country_idx in 1:length(countries)
        val = vals[country_idx, :, :]
        n = size(val, 1)

        pop = data.population[country_idx]
        num_total_days = data.num_total_days
        num_observed_days = length(data.cases[country_idx])

        country_name = countries[country_idx]

        # A tiny bit of preprocessing of the data
        preproc(x) = normalize_pop ? x ./ pop : x

        tmp = preproc(cumsum(val; dims = 1))
        qs = [quantile(tmp[t, :], [0.025, 0.5, 0.975]) for t = 1:n]
        lq, mq, hq = (eachrow(hcat(qs...))..., )

        push!(lqs, lq)
        push!(mqs, mq)
        push!(hqs, hq)
        push!(labels, country_name)
    end

    lqs = reduce(hcat, collect.(lqs))
    mqs = reduce(hcat, collect.(mqs))
    hqs = reduce(hcat, collect.(hqs))

    p = plot(; kwargs...)
    for country_idx in 1:length(countries)
        label = no_label ? "" : labels[country_idx]
        plot!(mqs[:, country_idx]; ribbon=(mqs[:, country_idx] - lqs[:, country_idx], hqs[:, country_idx] - mqs[:, country_idx]), label=label)
    end

    return p
end
```

    country_cumulative_prediction (generic function with 1 method)

{% enddetails %}

```julia
daily_deaths_arr = arrarrarr2arr(daily_deaths_posterior)
daily_deaths_arr = permutedims(daily_deaths_arr, (2, 3, 1))

daily_cases_arr = arrarrarr2arr(daily_cases_posterior)
daily_cases_arr = permutedims(daily_cases_arr, (2, 3, 1))
```

    14×100×4000 Array{Float64,3}:
    [:, :, 1] =
     31.6314   31.6314   31.6314   31.6314   …    156.61        151.178
     35.2791   35.2791   35.2791   35.2791       3646.08       3431.64
      5.82819   5.82819   5.82819   5.82819      6416.22       6282.52
     32.4507   32.4507   32.4507   32.4507       9749.19       9408.06
     33.8585   33.8585   33.8585   33.8585       6107.9        5812.33
     14.4027   14.4027   14.4027   14.4027   …  13849.1       13444.2
     13.4485   13.4485   13.4485   13.4485         73.396        70.6272
      5.32057   5.32057   5.32057   5.32057     45867.8       45212.0
     31.3424   31.3424   31.3424   31.3424        191.955       186.097
     93.6691   93.6691   93.6691   93.6691      53568.5       53719.7
     25.9993   25.9993   25.9993   25.9993   …    130.31        122.522
     63.4528   63.4528   63.4528   63.4528          0.135277      0.120549
     30.3088   30.3088   30.3088   30.3088         86.6885       82.4991
     58.7329   58.7329   58.7329   58.7329        364.967       341.81
    
    [:, :, 2] =
     21.8248   21.8248   21.8248   21.8248   …     59.4495        56.6651
     17.6305   17.6305   17.6305   17.6305       2751.21        2577.23
     17.5398   17.5398   17.5398   17.5398       2514.87        2422.85
     39.3773   39.3773   39.3773   39.3773       2572.54        2435.6
     13.8745   13.8745   13.8745   13.8745       7143.99        6796.27
      2.43846   2.43846   2.43846   2.43846  …  21861.8        21394.1
     16.6948   16.6948   16.6948   16.6948         12.4779        11.7615
      7.64907   7.64907   7.64907   7.64907     43911.9        43155.0
     54.9893   54.9893   54.9893   54.9893        164.217        158.063
     96.3331   96.3331   96.3331   96.3331      52492.8        52575.9
      6.69537   6.69537   6.69537   6.69537  …    291.955        278.593
     41.8925   41.8925   41.8925   41.8925          0.0549723      0.0484776
     10.4633   10.4633   10.4633   10.4633       1133.48        1110.4
     53.9036   53.9036   53.9036   53.9036        285.281        266.222
    
    [:, :, 3] =
      45.2317   45.2317   45.2317   45.2317  …     61.406         58.6976
      29.357    29.357    29.357    29.357       3176.11        2995.15
      30.394    30.394    30.394    30.394       1196.45        1139.83
     125.299   125.299   125.299   125.299       6239.52        5961.07
      36.4959   36.4959   36.4959   36.4959      4289.44        4052.54
      19.2367   19.2367   19.2367   19.2367  …  19999.4        19513.5
      16.405    16.405    16.405    16.405         13.0237        12.2817
      15.1937   15.1937   15.1937   15.1937     36906.7        36324.8
      47.9864   47.9864   47.9864   47.9864       198.327        192.355
     145.876   145.876   145.876   145.876      95512.4        93517.3
      12.7606   12.7606   12.7606   12.7606  …    140.283        132.076
      84.0577   84.0577   84.0577   84.0577         0.0887854      0.078702
      66.317    66.317    66.317    66.317         26.8103        25.0513
      50.1513   50.1513   50.1513   50.1513       385.437        362.086
    
    ...
    
    [:, :, 3998] =
     40.6005   40.6005   40.6005   40.6005   …     41.3201      39.257
     37.4406   37.4406   37.4406   37.4406       5168.87      4928.88
      6.66627   6.66627   6.66627   6.66627      1796.71      1724.79
     19.2496   19.2496   19.2496   19.2496       2893.68      2735.41
     21.6056   21.6056   21.6056   21.6056      13520.2      13034.5
      4.69097   4.69097   4.69097   4.69097  …  12729.8      12386.1
     42.8182   42.8182   42.8182   42.8182        326.627      322.394
      7.19116   7.19116   7.19116   7.19116     50203.4      49797.1
     26.3942   26.3942   26.3942   26.3942        263.88       256.656
     53.6565   53.6565   53.6565   53.6565      83072.5      82676.3
     24.4936   24.4936   24.4936   24.4936   …    153.229      144.346
     46.8848   46.8848   46.8848   46.8848          3.16499      2.94815
     74.0729   74.0729   74.0729   74.0729         34.9909      32.6986
     84.5948   84.5948   84.5948   84.5948        600.202      567.397
    
    [:, :, 3999] =
      17.5754    17.5754    17.5754    17.5754   …     246.31         239.819
      46.9688    46.9688    46.9688    46.9688        2641.87        2474.16
       5.1029     5.1029     5.1029     5.1029        2141.21        2071.62
      30.5983    30.5983    30.5983    30.5983        2429.53        2299.52
      15.1354    15.1354    15.1354    15.1354       15339.1        14856.2
      12.5852    12.5852    12.5852    12.5852   …   19716.9        19184.8
      20.7569    20.7569    20.7569    20.7569         198.359        193.751
       9.44519    9.44519    9.44519    9.44519      67919.1        66470.7
      46.0889    46.0889    46.0889    46.0889         106.309        102.061
      87.7028    87.7028    87.7028    87.7028      103695.0       102949.0
      34.3861    34.3861    34.3861    34.3861   …     299.303        285.163
     114.996    114.996    114.996    114.996            0.367775       0.332197
      51.0231    51.0231    51.0231    51.0231         306.67         296.26
      95.074     95.074     95.074     95.074          683.525        647.794
    
    [:, :, 4000] =
     41.9978   41.9978   41.9978   41.9978   …    267.147        260.315
     29.6776   29.6776   29.6776   29.6776       6149.86        5851.81
     45.4262   45.4262   45.4262   45.4262       2766.05        2682.29
     67.3271   67.3271   67.3271   67.3271       3339.49        3174.33
     72.374    72.374    72.374    72.374       13353.4        12889.1
      5.65709   5.65709   5.65709   5.65709  …  39438.8        38904.1
     17.4738   17.4738   17.4738   17.4738         49.7947        47.6519
     16.4718   16.4718   16.4718   16.4718      33295.2        32990.2
     65.9772   65.9772   65.9772   65.9772        199.27         193.239
     87.5592   87.5592   87.5592   87.5592      32651.7        32673.7
     11.5608   11.5608   11.5608   11.5608   …    480.357        461.34
     88.1057   88.1057   88.1057   88.1057          0.0795788      0.070559
     27.258    27.258    27.258    27.258          97.7505        92.8548
     28.6621   28.6621   28.6621   28.6621        577.574        545.546

To make the plots interactive, we're going to use the [`PlotlyJS.jl`](https://github.com/sglyon/PlotlyJS.jl) backend (which generates [plotly.js](https://github.com/plotly/plotly.js/) plots under the hood):

```julia
plotlyjs()
```

    Plots.PlotlyJSBackend()

```julia
country_cumulative_prediction(daily_deaths_arr; size = (800, 300))
title!("Expected deaths (95% intervals)")
```

{% include plotly.html id='plot1' json='../assets/figures/2020-05-04-Imperial-Report13-analysis/plot1.json' %}

```julia
country_cumulative_prediction(daily_deaths_arr; normalize_pop = true, size = (800, 300))
title!("Expected deaths / population (95% intervals)")
```

{% include plotly.html id='plot2' json='../assets/figures/2020-05-04-Imperial-Report13-analysis/plot2.json' %}

In the following sections we're instead going to use `PyPlot.jl` as backend (which uses Python's `matplotlib` under the hood) instead of `Plotly.js` since too many `Plotly.js` plots can slow down even the finest of computers.

```julia
pyplot()
```

    Plots.PyPlotBackend()


# Prior

Before we do any inference it can be useful to inspect the *prior* distribution, in particular if you are working with a hierarchical model where the dependencies in the prior might lead to some unexpected behavior. In Turing.jl you can sample a chain from the prior using `sample`, much in the same way as you would sample from the posterior.

```julia
chain_prior = sample(m, Turing.Inference.PriorSampler(), 1_000);
```

```julia
plot(chain_prior[[:ϕ, :τ, :κ]]; α = .5, linewidth=1.5)
```

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/uk-prior-kappa-phi-tau-sample-plot.png)

For the same reasons it can be very useful to inspect the *prior predictive* distribution.

```julia
# Compute the "generated quantities" for the PRIOR
generated_prior = vectup2tupvec(generated_quantities(m, chain_prior));
daily_cases_prior, daily_deaths_prior, Rt_prior, Rt_adj_prior = generated_prior;
```

```julia
country_prediction_plot(uk_index, daily_cases_prior, daily_deaths_prior, Rt_prior; main_title = "(prior)")
```

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/uk-predictive-prior-Rt.png)

And with the Rt *adjusted for remaining population*:

```julia
country_prediction_plot(uk_index, daily_cases_prior, daily_deaths_prior, Rt_adj_prior; main_title = "(prior)")
```

At this point it might be useful to remind ourselves of the total population of UK is:

```julia
data.population[uk_index]
```

    67886004

As we can see from the figures, the prior allows scenarios such as

-   *all* of the UK being infected
-   effects of interventions, e.g. `lockdown`, having a *negative* effect on `Rt` (in the sense that it can actually *increase* the spread of the virus); you can see this from the fact that the 95% confidence interval widens after one of the interventions

But at the same time, it's clear that a very sudden increase from 0% to 100% of the population being infected is almost impossible under the prior. All in all, the model prior seems a reasonable choice: it allows for extreme situations without putting too much probabilty "mass" on those, while still encoding some structure in the model.


# Posterior inference

```julia
parameters = (
    warmup = 1000,
    steps = 3000
);
```


## Inference


### Run

To perform inference for the model we would simply run the code below:

```julia
chains_posterior = sample(m_no_pred, NUTS(parameters.warmup, 0.95, 10), parameters.steps + parameters.warmup)
```

*But* unfortunately it takes quite a while to run. Performing inference using `NUTS` with `1000` steps for adaptation/warmup and `3000` sample steps takes ~2hrs on a 6-core computer with `JULIA_NUM_THREADS = 6`. And so we're instead going to load in the chains needed.

In contrast, `Stan` only takes roughly 1hr *on a single thread* using the base model from the repository. On a single thread `Turing.jl` is ~4-5X slower for this model, which is quite signficant.

This generally means that if you have a clear model in mind (or you're already very familiar with `Stan`), you probably want to use `Stan` for these kind of models. On the other hand, if you're in the process of heavily tweaking your model and need to be able to do `m = model(data...); m()` to check if it works, or you want more flexibility in what you can do with your model, e.g. discrete variables, `Turing.jl` might be a good option.

And this is an additional reason why we wanted to perform this replication study: we want `Turing.jl` to be *useful* and the way to check this is by applying `Turing.jl` to real-world problem rather than *just* benchmarks (though those are important too).

And regarding the performance difference, it really comes down to the difference in implementation of automatic differentation (AD). `Turing.jl` allows you to choose from the goto AD packages in Julia, e.g. in our runs we used [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl), while `Stan` as a very, very fast AD implementation written exclusively for `Stan`. This difference becomes very clear in models such as this one where we have a lot of for-loops and recursive relationships (because this means that we can't easily vectorize). For-loops in Julia are generally blazingly fast, but with AD there's a bit of overhead. But that also means that in `Turing.jl` you have the ability to choose between different approaches to AD, i.e. forward-mode or reverse-mode, each with their different tradeoffs, and thus will benefit from potentially interesting future work, e.g. source-to-source AD using [Zygote.jl](https://github.com/FluxML/Zygote.jl).[^fn3]

And one interesting tidbit is that you can very easily use `pystan` within `PyCall.jl` to sample from a `Stan` model, and then convert the results into a [MCMCChains.jl](https://github.com/TuringLang/MCMCChains.jl). This has some nice implications:

-   we can use all the convenient posterior analysis tools available in MCMCChains.jl to analyze chains from `Stan`
-   we can use the `generated_quantities` method in this notebook to execute the `Turing.jl` `Model` on the samples obtain using `Stan`

This was quite useful for us to be able validate the results from `Turing.jl` against those from `Stan`, and made it very easy to check that indeed `Turing.jl` and `Stan` produce the same results. You can find examples of this in the [notebooks in our repository](https://github.com/TuringLang/Covid19).

{% details Sampling from `Stan` in Julia using `pystan` %}

First we import `PyCall`, allowing us to call Python code from within Julia.

```julia
using PyCall

using PyCall: pyimport
pystan = pyimport("pystan");
```

Then we define the Stan model as a string

```julia
model_str = raw"""
data {
  int <lower=1> M; // number of countries
  int <lower=1> P; // number of covariates
  int <lower=1> N0; // number of days for which to impute infections
  int<lower=1> N[M]; // days of observed data for country m. each entry must be <= N2
  int<lower=1> N2; // days of observed data + # of days to forecast
  int cases[N2,M]; // reported cases
  int deaths[N2, M]; // reported deaths -- the rows with i > N contain -1 and should be ignored
  matrix[N2, M] f; // h * s
  matrix[N2, P] X[M]; // features matrix
  int EpidemicStart[M];
  real pop[M];
  real SI[N2]; // fixed pre-calculated SI using emprical data from Neil
}

transformed data {
  vector[N2] SI_rev; // SI in reverse order
  vector[N2] f_rev[M]; // f in reversed order

  for(i in 1:N2)
    SI_rev[i] = SI[N2-i+1];

  for(m in 1:M){
    for(i in 1:N2) {
     f_rev[m, i] = f[N2-i+1,m];
    }
  }
}


parameters {
  real<lower=0> mu[M]; // intercept for Rt
  real<lower=0> alpha_hier[P]; // sudo parameter for the hier term for alpha
  real<lower=0> gamma;
  vector[M] lockdown;
  real<lower=0> kappa;
  real<lower=0> y[M];
  real<lower=0> phi;
  real<lower=0> tau;
  real <lower=0> ifr_noise[M];
}

transformed parameters {
    vector[P] alpha;
    matrix[N2, M] prediction = rep_matrix(0,N2,M);
    matrix[N2, M] E_deaths  = rep_matrix(0,N2,M);
    matrix[N2, M] Rt = rep_matrix(0,N2,M);
    matrix[N2, M] Rt_adj = Rt;

    {
      matrix[N2,M] cumm_sum = rep_matrix(0,N2,M);
      for(i in 1:P){
        alpha[i] = alpha_hier[i] - ( log(1.05) / 6.0 );
      }
      for (m in 1:M){
        prediction[1:N0,m] = rep_vector(y[m],N0); // learn the number of cases in the first N0 days
        cumm_sum[2:N0,m] = cumulative_sum(prediction[2:N0,m]);

        Rt[,m] = mu[m] * exp(-X[m] * alpha - X[m][,5] * lockdown[m]);
        Rt_adj[1:N0,m] = Rt[1:N0,m];
        for (i in (N0+1):N2) {
          real convolution = dot_product(sub_col(prediction, 1, m, i-1), tail(SI_rev, i-1));
          cumm_sum[i,m] = cumm_sum[i-1,m] + prediction[i-1,m];
          Rt_adj[i,m] = ((pop[m]-cumm_sum[i,m]) / pop[m]) * Rt[i,m];
          prediction[i, m] = Rt_adj[i,m] * convolution;
        }
        E_deaths[1, m]= 1e-15 * prediction[1,m];
        for (i in 2:N2){
          E_deaths[i,m] = ifr_noise[m] * dot_product(sub_col(prediction, 1, m, i-1), tail(f_rev[m], i-1));
        }
      }
    }
}
model {
  tau ~ exponential(0.03);
  for (m in 1:M){
      y[m] ~ exponential(1/tau);
  }
  gamma ~ normal(0,.2);
  lockdown ~ normal(0,gamma);
  phi ~ normal(0,5);
  kappa ~ normal(0,0.5);
  mu ~ normal(3.28, kappa); // citation: https://academic.oup.com/jtm/article/27/2/taaa021/5735319
  alpha_hier ~ gamma(.1667,1);
  ifr_noise ~ normal(1,0.1);
  for(m in 1:M){
    deaths[EpidemicStart[m]:N[m], m] ~ neg_binomial_2(E_deaths[EpidemicStart[m]:N[m], m], phi);
   }
}

generated quantities {
    matrix[N2, M] prediction0 = rep_matrix(0,N2,M);
    matrix[N2, M] E_deaths0  = rep_matrix(0,N2,M);

    {
      matrix[N2,M] cumm_sum0 = rep_matrix(0,N2,M);
      for (m in 1:M){
         for (i in 2:N0){
          cumm_sum0[i,m] = cumm_sum0[i-1,m] + y[m]; 
        }
        prediction0[1:N0,m] = rep_vector(y[m],N0); 
        for (i in (N0+1):N2) {
          real convolution0 = dot_product(sub_col(prediction0, 1, m, i-1), tail(SI_rev, i-1));
          cumm_sum0[i,m] = cumm_sum0[i-1,m] + prediction0[i-1,m];
          prediction0[i, m] = ((pop[m]-cumm_sum0[i,m]) / pop[m]) * mu[m] * convolution0;
        }
        E_deaths0[1, m]= 1e-15 * prediction0[1,m];
        for (i in 2:N2){
          E_deaths0[i,m] = ifr_noise[m] * dot_product(sub_col(prediction0, 1, m, i-1), tail(f_rev[m], i-1));
        }
      }
    }
}
"""
```

We need the data to be in a format compatible with the Stan model, which we can accomplish by converting `rdata` into a `Dict`:

```julia
d = Dict([(k, rdata[k]) for k in keys(rdata)]); # `values(df)` and `keys(df)` have different ordering so DON'T do `Dict(keys(df), values(df))`
```

Then we can compile the `Stan` model

```julia
sm = pystan.StanModel(model_code=model_str)
```

And finally fit:

```julia
fit_stan(n_iters=300, warmup=100) = sm.sampling(
    data=d, iter=n_iters, chains=1, warmup=warmup, algorithm="NUTS", 
    control=Dict(
        "adapt_delta" => 0.95,
        "max_treedepth" => 10
    )
)
f = fit_stan(parameters.steps + parameters.warmup, parameters.warmup)
```

From the fit we can extract the inferred parameters:

```julia
la = f.extract(permuted=true)
```

Or, if you've done this before and saved the results using `Serialization.jl`, you can load the results:

```julia
using Serialization

stan_chain_fname = first([s for s in readdir(outdir()) if occursin("stan", s)])
la = open(io -> deserialize(io), outdir(stan_chain_fname), "r")
```

    Dict{Any,Any} with 17 entries:
      "E_deaths"    => [6.52866e-14 5.9334e-7 … 2.40106 2.31387; 4.77975e-14 3.5296…
      "ifr_noise"   => [0.890352 1.05228 … 1.19123 0.833645; 0.723459 1.03243 … 0.8…
      "alpha"       => [-0.00813169 -0.00812908 … 0.723476 -0.00112684; 0.237363 0.…
      "E_deaths0"   => [6.52866e-14 5.9334e-7 … 83.0411 71.0609; 4.77975e-14 3.5296…
      "tau"         => [53.4179, 96.0101, 38.1312, 68.78, 49.8292, 45.3714, 80.3358…
      "Rt_adj"      => [3.69848 3.69848 … 0.762185 0.762171; 4.19599 4.19599 … 0.77…
      "mu"          => [3.69848 3.8664 … 3.68493 4.11784; 4.19599 3.8219 … 4.09925 …
      "prediction0" => [65.2866 65.2866 … 1.56204 1.22123; 47.7975 47.7975 … 0.1296…
      "lp__"        => [560668.0, 560680.0, 560685.0, 560670.0, 560671.0, 560694.0,…
      "kappa"       => [1.05947, 0.913342, 1.05901, 1.17966, 1.12473, 1.21213, 0.99…
      "alpha_hier"  => [1.73288e-16 2.6161e-6 … 0.731608 0.00700485; 0.245495 0.023…
      "prediction"  => [65.2866 65.2866 … 106.091 102.225; 47.7975 47.7975 … 206.13…
      "phi"         => [6.8105, 7.36518, 6.41771, 7.13139, 6.70706, 7.06452, 7.0425…
      "gamma"       => [0.107149, 0.129599, 0.118666, 0.274338, 0.179847, 0.0386703…
      "lockdown"    => [-0.0199127 0.0841175 … -0.0854961 0.245588; 0.00700599 0.02…
      "Rt"          => [3.69848 3.69848 … 0.770378 0.770378; 4.19599 4.19599 … 0.78…
      "y"           => [65.2866 22.1974 … 75.7377 63.0177; 47.7975 27.7832 … 48.400…

And if we want to compare it with the results from `Turing.jl` it can be convenient to rename some of the variables

```julia
rename!(
    la,
    "alpha" => "α",
    "alpha_hier" => "α_hier",
    "kappa" => "κ",
    "gamma" => "γ",
    "mu" => "μ",
    "phi" => "ϕ",
    "tau" => "τ"
)
```

    Dict{Any,Any} with 17 entries:
      "α_hier"      => [1.73288e-16 2.6161e-6 … 0.731608 0.00700485; 0.245495 0.023…
      "μ"           => [3.69848 3.8664 … 3.68493 4.11784; 4.19599 3.8219 … 4.09925 …
      "E_deaths"    => [6.52866e-14 5.9334e-7 … 2.40106 2.31387; 4.77975e-14 3.5296…
      "ifr_noise"   => [0.890352 1.05228 … 1.19123 0.833645; 0.723459 1.03243 … 0.8…
      "ϕ"           => [6.8105, 7.36518, 6.41771, 7.13139, 6.70706, 7.06452, 7.0425…
      "E_deaths0"   => [6.52866e-14 5.9334e-7 … 83.0411 71.0609; 4.77975e-14 3.5296…
      "α"           => [-0.00813169 -0.00812908 … 0.723476 -0.00112684; 0.237363 0.…
      "κ"           => [1.05947, 0.913342, 1.05901, 1.17966, 1.12473, 1.21213, 0.99…
      "Rt_adj"      => [3.69848 3.69848 … 0.762185 0.762171; 4.19599 4.19599 … 0.77…
      "γ"           => [0.107149, 0.129599, 0.118666, 0.274338, 0.179847, 0.0386703…
      "prediction0" => [65.2866 65.2866 … 1.56204 1.22123; 47.7975 47.7975 … 0.1296…
      "τ"           => [53.4179, 96.0101, 38.1312, 68.78, 49.8292, 45.3714, 80.3358…
      "lp__"        => [560668.0, 560680.0, 560685.0, 560670.0, 560671.0, 560694.0,…
      "prediction"  => [65.2866 65.2866 … 106.091 102.225; 47.7975 47.7975 … 206.13…
      "lockdown"    => [-0.0199127 0.0841175 … -0.0854961 0.245588; 0.00700599 0.02…
      "Rt"          => [3.69848 3.69848 … 0.770378 0.770378; 4.19599 4.19599 … 0.78…
      "y"           => [65.2866 22.1974 … 75.7377 63.0177; 47.7975 27.7832 … 48.400…

```julia
# Extract a subset of the variables, since we don't want everything in a `Chains` object
la_subset = Dict(
    k => la[k] for k in 
    ["y", "κ", "α_hier", "ϕ", "τ", "ifr_noise", "μ", "γ", "lockdown"]
)
```

    Dict{String,Array{Float64,N} where N} with 9 entries:
      "α_hier"    => [1.73288e-16 2.6161e-6 … 0.731608 0.00700485; 0.245495 0.02364…
      "μ"         => [3.69848 3.8664 … 3.68493 4.11784; 4.19599 3.8219 … 4.09925 3.…
      "γ"         => [0.107149, 0.129599, 0.118666, 0.274338, 0.179847, 0.0386703, …
      "τ"         => [53.4179, 96.0101, 38.1312, 68.78, 49.8292, 45.3714, 80.3358, …
      "ϕ"         => [6.8105, 7.36518, 6.41771, 7.13139, 6.70706, 7.06452, 7.04253,…
      "ifr_noise" => [0.890352 1.05228 … 1.19123 0.833645; 0.723459 1.03243 … 0.892…
      "κ"         => [1.05947, 0.913342, 1.05901, 1.17966, 1.12473, 1.21213, 0.9978…
      "lockdown"  => [-0.0199127 0.0841175 … -0.0854961 0.245588; 0.00700599 0.0258…
      "y"         => [65.2866 22.1974 … 75.7377 63.0177; 47.7975 27.7832 … 48.4006 …

In `Covid19.jl` we've added a constructor for `MCMCChains.Chains` which takes a `Dict` as an argument, easily allowing us to convert the `la` from `Stan` into a `Chains` object.

```julia
function MCMCChains._cat(::Val{3}, c1::Chains, args::Chains...)
    # check inputs
    rng = range(c1)
    # (OR not, he he he)
    # all(c -> range(c) == rng, args) || throw(ArgumentError("chain ranges differ"))
    nms = names(c1)
    all(c -> names(c) == nms, args) || throw(ArgumentError("chain names differ"))

    # concatenate all chains
    data = mapreduce(c -> c.value.data, (x, y) -> cat(x, y; dims = 3), args;
                     init = c1.value.data)
    value = MCMCChains.AxisArray(data; iter = rng, var = nms, chain = 1:size(data, 3))

    return Chains(value, missing, c1.name_map, c1.info)
end
```

```julia
stan_chain = Chains(la_subset); # <= results in all chains being concatenated together so we need to manually "separate" them

steps_per_chain = parameters.steps
num_chains = Int(length(stan_chain) // steps_per_chain)

stan_chains = [stan_chain[1 + (i - 1) * steps_per_chain:i * steps_per_chain] for i = 1:num_chains];
stan_chains = chainscat(stan_chains...);
stan_chains = stan_chains[1:3:end] # thin
```

    Object of type Chains, with data of type 1000×66×4 Array{Float64,3}
    
    Iterations        = 1:2998
    Thinning interval = 3
    Chains            = 1, 2, 3, 4
    Samples per chain = 1000
    parameters        = α_hier[1], α_hier[2], α_hier[3], α_hier[4], α_hier[5], α_hier[6], μ[1], μ[2], μ[3], μ[4], μ[5], μ[6], μ[7], μ[8], μ[9], μ[10], μ[11], μ[12], μ[13], μ[14], γ, τ, ϕ, ifr_noise[1], ifr_noise[2], ifr_noise[3], ifr_noise[4], ifr_noise[5], ifr_noise[6], ifr_noise[7], ifr_noise[8], ifr_noise[9], ifr_noise[10], ifr_noise[11], ifr_noise[12], ifr_noise[13], ifr_noise[14], κ, lockdown[1], lockdown[2], lockdown[3], lockdown[4], lockdown[5], lockdown[6], lockdown[7], lockdown[8], lockdown[9], lockdown[10], lockdown[11], lockdown[12], lockdown[13], lockdown[14], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14]
    
    2-element Array{ChainDataFrame,1}
    
    Summary Statistics
         parameters      mean      std  naive_se    mcse        ess   r_hat
      ─────────────  ────────  ───────  ────────  ──────  ─────────  ──────
          α_hier[1]    0.0250   0.0509    0.0008  0.0009  3762.6907  0.9994
          α_hier[2]    0.0294   0.0562    0.0009  0.0010  4096.3296  1.0021
          α_hier[3]    0.4865   0.1666    0.0026  0.0031  3769.7698  1.0021
          α_hier[4]    0.0150   0.0322    0.0005  0.0005  3963.6029  1.0003
          α_hier[5]    1.1267   0.1498    0.0024  0.0022  3990.6714  0.9996
          α_hier[6]    0.0647   0.1037    0.0016  0.0024  4037.5795  1.0053
               μ[1]    3.9594   0.4722    0.0075  0.0075  4064.2020  0.9996
               μ[2]    3.5951   0.1701    0.0027  0.0029  3684.0289  1.0005
               μ[3]    4.2405   0.4080    0.0065  0.0092  4232.5667  1.0054
               μ[4]    4.5359   0.3732    0.0059  0.0078  4075.5280  1.0031
               μ[5]    3.9081   0.2873    0.0045  0.0076  4106.5761  1.0080
               μ[6]    4.9298   0.3241    0.0051  0.0050  3850.5272  1.0009
               μ[7]    3.8319   0.5364    0.0085  0.0075  3956.2449  0.9997
               μ[8]    6.3026   0.7068    0.0112  0.0106  3641.7136  1.0005
               μ[9]    4.3273   0.5181    0.0082  0.0077  4059.0140  1.0002
              μ[10]    2.4113   0.2297    0.0036  0.0049  4080.2115  1.0035
              μ[11]    3.8111   0.3755    0.0059  0.0058  3906.9576  0.9996
              μ[12]    2.0742   0.2940    0.0046  0.0048  4350.2472  0.9997
              μ[13]    3.9327   0.4426    0.0070  0.0068  4062.9915  0.9999
              μ[14]    3.6872   0.3501    0.0055  0.0052  3971.6142  1.0001
                  γ    0.1132   0.0683    0.0011  0.0014  3980.7969  1.0015
                  τ   54.4981  17.5050    0.2768  0.2810  3759.2791  1.0007
                  ϕ    6.8784   0.5566    0.0088  0.0078  4017.5627  1.0000
       ifr_noise[1]    0.9994   0.1006    0.0016  0.0014  4337.3600  1.0002
       ifr_noise[2]    0.9957   0.0996    0.0016  0.0018  3229.4271  1.0008
       ifr_noise[3]    0.9927   0.1001    0.0016  0.0013  4205.8756  0.9997
       ifr_noise[4]    1.0019   0.0981    0.0016  0.0016  3779.6148  0.9994
       ifr_noise[5]    0.9973   0.1002    0.0016  0.0015  4230.6973  1.0002
       ifr_noise[6]    0.9901   0.0997    0.0016  0.0014  4019.0526  0.9992
       ifr_noise[7]    0.9977   0.0985    0.0016  0.0019  4102.5786  0.9997
       ifr_noise[8]    0.9926   0.0989    0.0016  0.0014  4172.9264  0.9998
       ifr_noise[9]    1.0024   0.0998    0.0016  0.0015  4036.0007  1.0003
      ifr_noise[10]    1.0143   0.1004    0.0016  0.0020  3965.4960  1.0003
      ifr_noise[11]    0.9940   0.0984    0.0016  0.0014  4248.2237  0.9999
      ifr_noise[12]    1.0104   0.0978    0.0015  0.0016  4037.6589  1.0012
      ifr_noise[13]    0.9997   0.0999    0.0016  0.0016  3775.1249  1.0007
      ifr_noise[14]    1.0048   0.0990    0.0016  0.0016  4019.4484  0.9996
                  κ    1.1204   0.2167    0.0034  0.0029  4136.1687  0.9996
        lockdown[1]    0.0034   0.1096    0.0017  0.0012  4217.7275  0.9991
        lockdown[2]   -0.0194   0.0852    0.0013  0.0014  4045.8681  1.0009
        lockdown[3]   -0.0171   0.1053    0.0017  0.0019  4125.5208  1.0011
        lockdown[4]    0.1309   0.1242    0.0020  0.0023  4149.1488  1.0018
        lockdown[5]    0.0144   0.1114    0.0018  0.0020  4275.9326  1.0002
        lockdown[6]   -0.0287   0.0933    0.0015  0.0012  4116.3446  1.0001
        lockdown[7]   -0.0197   0.1247    0.0020  0.0021  4076.6525  1.0002
        lockdown[8]   -0.0800   0.1130    0.0018  0.0019  3866.1604  1.0006
        lockdown[9]   -0.0104   0.1047    0.0017  0.0016  4021.3347  0.9997
       lockdown[10]    0.0026   0.1326    0.0021  0.0023  4007.4693  0.9997
       lockdown[11]    0.0121   0.1079    0.0017  0.0019  3829.4277  0.9996
       lockdown[12]    0.0030   0.1271    0.0020  0.0018  4228.0928  1.0003
       lockdown[13]   -0.0019   0.1141    0.0018  0.0016  4215.2192  0.9997
       lockdown[14]    0.0548   0.1179    0.0019  0.0020  3988.7643  1.0006
               y[1]   47.1930  23.4949    0.3715  0.3560  4301.2466  0.9999
               y[2]   39.9589  13.3690    0.2114  0.2264  3840.8341  1.0004
               y[3]   20.2291   9.8615    0.1559  0.1618  3984.8827  1.0001
               y[4]   74.6538  33.9382    0.5366  0.6232  4021.1495  1.0018
               y[5]   44.3600  17.6755    0.2795  0.3848  3408.4552  1.0043
               y[6]    9.5225   4.5031    0.0712  0.0750  3902.1911  1.0008
               y[7]   30.0746  17.9098    0.2832  0.2778  3956.5131  1.0009
               y[8]   17.3519  11.2451    0.1778  0.1606  3384.7290  0.9995
               y[9]   75.4817  36.1157    0.5710  0.5745  4155.4331  1.0010
              y[10]  144.5704  54.9286    0.8685  0.9158  4175.9150  1.0001
              y[11]   26.1128  13.4270    0.2123  0.2135  3739.8586  0.9996
              y[12]   84.6856  44.8467    0.7091  0.6679  4167.7817  0.9995
              y[13]   55.2918  25.9756    0.4107  0.4109  4327.8064  0.9999
              y[14]   83.2622  36.3821    0.5753  0.5105  3890.7244  1.0000
    
    Quantiles
         parameters     2.5%     25.0%     50.0%     75.0%     97.5%
      ─────────────  ───────  ────────  ────────  ────────  ────────
          α_hier[1]   0.0000    0.0000    0.0018    0.0244    0.1820
          α_hier[2]   0.0000    0.0000    0.0024    0.0313    0.2057
          α_hier[3]   0.0946    0.3853    0.4948    0.5975    0.7926
          α_hier[4]   0.0000    0.0000    0.0010    0.0126    0.1224
          α_hier[5]   0.8434    1.0235    1.1289    1.2272    1.4198
          α_hier[6]   0.0000    0.0001    0.0103    0.0903    0.3542
               μ[1]   3.1090    3.6397    3.9181    4.2532    4.9781
               μ[2]   3.2701    3.4824    3.5906    3.7117    3.9333
               μ[3]   3.5392    3.9578    4.1962    4.4800    5.1477
               μ[4]   3.9023    4.2706    4.4931    4.7583    5.3515
               μ[5]   3.4586    3.7031    3.8641    4.0777    4.5887
               μ[6]   4.3074    4.7095    4.9339    5.1434    5.5663
               μ[7]   2.8756    3.4622    3.8022    4.1684    4.9753
               μ[8]   4.9469    5.8267    6.2903    6.7636    7.7464
               μ[9]   3.3888    3.9792    4.2975    4.6499    5.4316
              μ[10]   2.0541    2.2494    2.3785    2.5382    2.9642
              μ[11]   3.1351    3.5524    3.7874    4.0452    4.5922
              μ[12]   1.5522    1.8702    2.0543    2.2570    2.7023
              μ[13]   3.1392    3.6240    3.9055    4.2166    4.8550
              μ[14]   3.0547    3.4411    3.6641    3.9070    4.4457
                  γ   0.0106    0.0621    0.1066    0.1540    0.2669
                  τ  28.3554   41.8978   51.6206   64.4602   95.8893
                  ϕ   5.8450    6.4945    6.8693    7.2375    8.0241
       ifr_noise[1]   0.8053    0.9319    0.9994    1.0669    1.1991
       ifr_noise[2]   0.8027    0.9289    0.9982    1.0624    1.1919
       ifr_noise[3]   0.8002    0.9254    0.9940    1.0584    1.1875
       ifr_noise[4]   0.8146    0.9355    1.0002    1.0669    1.1992
       ifr_noise[5]   0.8044    0.9283    0.9971    1.0648    1.1926
       ifr_noise[6]   0.7977    0.9221    0.9916    1.0580    1.1871
       ifr_noise[7]   0.7976    0.9337    0.9976    1.0632    1.1926
       ifr_noise[8]   0.7980    0.9262    0.9929    1.0585    1.1868
       ifr_noise[9]   0.8115    0.9365    1.0023    1.0712    1.1955
      ifr_noise[10]   0.8129    0.9487    1.0128    1.0832    1.2092
      ifr_noise[11]   0.7985    0.9277    0.9939    1.0599    1.1874
      ifr_noise[12]   0.8173    0.9461    1.0107    1.0743    1.2046
      ifr_noise[13]   0.8069    0.9315    1.0002    1.0663    1.2020
      ifr_noise[14]   0.8168    0.9357    1.0049    1.0718    1.1968
                  κ   0.7446    0.9697    1.1048    1.2558    1.5970
        lockdown[1]  -0.2211   -0.0529    0.0012    0.0552    0.2476
        lockdown[2]  -0.2064   -0.0681   -0.0127    0.0270    0.1537
        lockdown[3]  -0.2501   -0.0689   -0.0097    0.0368    0.1982
        lockdown[4]  -0.0409    0.0309    0.1091    0.2083    0.4163
        lockdown[5]  -0.2099   -0.0406    0.0063    0.0678    0.2665
        lockdown[6]  -0.2389   -0.0807   -0.0173    0.0255    0.1494
        lockdown[7]  -0.2983   -0.0748   -0.0083    0.0442    0.2140
        lockdown[8]  -0.3452   -0.1409   -0.0560   -0.0025    0.0978
        lockdown[9]  -0.2333   -0.0598   -0.0036    0.0415    0.2027
       lockdown[10]  -0.2747   -0.0594    0.0012    0.0615    0.2855
       lockdown[11]  -0.2111   -0.0428    0.0049    0.0654    0.2543
       lockdown[12]  -0.2539   -0.0534    0.0010    0.0600    0.2765
       lockdown[13]  -0.2374   -0.0579   -0.0012    0.0530    0.2304
       lockdown[14]  -0.1415   -0.0145    0.0319    0.1134    0.3319
               y[1]  15.0725   30.7683   42.5110   58.6852  108.5279
               y[2]  20.0808   30.2341   38.0373   47.2979   71.8859
               y[3]   7.0457   13.3609   18.2912   24.7683   45.4752
               y[4]  25.4333   50.1704   68.8542   92.8525  155.6052
               y[5]  17.5663   31.6674   41.9898   54.2124   85.7669
               y[6]   3.5671    6.3678    8.5704   11.6961   20.4845
               y[7]   7.6940   17.3636   25.8101   37.9979   74.9977
               y[8]   4.6864    9.8616   14.5959   21.2624   47.3559
               y[9]  27.8821   50.0205   68.1117   93.5616  165.9393
              y[10]  59.6325  104.7132  136.6978  177.1941  270.4725
              y[11]   8.3990   16.7150   23.4562   32.1587   59.1546
              y[12]  25.3671   53.5384   74.8896  106.1230  199.7460
              y[13]  19.4379   36.8008   50.4436   68.4129  119.8110
              y[14]  30.3896   57.3074   76.9474  101.9789  170.3966

{% enddetails %}


### Load

Unfortunately the resulting chains, each with 3000 steps, take up a fair bit of space and are thus too large to include in the Github repository. As a temporary hack around this, you can find download the chains from [this link](https://drive.google.com/open?id=16PomGVnjPI1Q4KLdA9gRloVolfRmrhPP). Then you simply navigate to the project-directory and unpack.

With that, we can load the chains from disk:

```julia
filenames = [
    relpath(outdir(s)) for s in readdir(outdir())
    if occursin(savename(parameters), s) && occursin("seed", s)
]
length(filenames)
```

    4

```julia
chains_posterior_vec = [read(fname, Chains) for fname in filenames]; # Read the different chains
chains_posterior = chainscat(chains_posterior_vec...); # Concatenate them
chains_posterior = chains_posterior[1:3:end] # <= Thin so we're left with 1000 samples
```

    Object of type Chains, with data of type 1000×78×4 Array{Float64,3}
    
    Iterations        = 1:2998
    Thinning interval = 3
    Chains            = 1, 2, 3, 4
    Samples per chain = 1000
    internals         = acceptance_rate, hamiltonian_energy, hamiltonian_energy_error, is_accept, log_density, lp, max_hamiltonian_energy_error, n_steps, nom_step_size, numerical_error, step_size, tree_depth
    parameters        = ifr_noise[1], ifr_noise[2], ifr_noise[3], ifr_noise[4], ifr_noise[5], ifr_noise[6], ifr_noise[7], ifr_noise[8], ifr_noise[9], ifr_noise[10], ifr_noise[11], ifr_noise[12], ifr_noise[13], ifr_noise[14], lockdown[1], lockdown[2], lockdown[3], lockdown[4], lockdown[5], lockdown[6], lockdown[7], lockdown[8], lockdown[9], lockdown[10], lockdown[11], lockdown[12], lockdown[13], lockdown[14], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14], α_hier[1], α_hier[2], α_hier[3], α_hier[4], α_hier[5], α_hier[6], γ, κ, μ[1], μ[2], μ[3], μ[4], μ[5], μ[6], μ[7], μ[8], μ[9], μ[10], μ[11], μ[12], μ[13], μ[14], τ, ϕ
    
    2-element Array{ChainDataFrame,1}
    
    Summary Statistics
         parameters      mean      std  naive_se    mcse        ess   r_hat
      ─────────────  ────────  ───────  ────────  ──────  ─────────  ──────
       ifr_noise[1]    0.9993   0.1018    0.0016  0.0016  4000.7654  1.0007
       ifr_noise[2]    1.0000   0.0985    0.0016  0.0016  4130.0153  0.9994
       ifr_noise[3]    0.9943   0.0986    0.0016  0.0017  4009.9552  0.9995
       ifr_noise[4]    1.0014   0.0996    0.0016  0.0019  4073.9127  1.0006
       ifr_noise[5]    0.9975   0.1017    0.0016  0.0017  3997.8443  1.0000
       ifr_noise[6]    0.9919   0.0971    0.0015  0.0013  4183.1643  0.9994
       ifr_noise[7]    0.9935   0.1000    0.0016  0.0016  3754.0585  1.0013
       ifr_noise[8]    0.9906   0.1019    0.0016  0.0019  3764.0323  1.0002
       ifr_noise[9]    1.0022   0.0983    0.0016  0.0017  3271.0573  1.0000
      ifr_noise[10]    1.0143   0.0979    0.0015  0.0015  3713.8604  0.9999
      ifr_noise[11]    0.9958   0.0990    0.0016  0.0015  4070.7331  1.0007
      ifr_noise[12]    1.0031   0.1001    0.0016  0.0015  4062.0625  0.9994
      ifr_noise[13]    1.0020   0.0983    0.0016  0.0017  3878.4554  1.0003
      ifr_noise[14]    1.0051   0.0977    0.0015  0.0016  3411.2449  1.0003
        lockdown[1]   -0.0015   0.1038    0.0016  0.0017  3351.5301  0.9992
        lockdown[2]   -0.0189   0.0837    0.0013  0.0016  2301.1025  1.0007
        lockdown[3]   -0.0172   0.0994    0.0016  0.0018  3307.7319  1.0002
        lockdown[4]    0.1181   0.1197    0.0019  0.0047   755.5366  1.0098
        lockdown[5]    0.0104   0.1056    0.0017  0.0021  2241.6092  1.0013
        lockdown[6]   -0.0276   0.0903    0.0014  0.0017  1854.2957  1.0007
        lockdown[7]   -0.0183   0.1154    0.0018  0.0022  3101.1043  1.0002
        lockdown[8]   -0.0733   0.1121    0.0018  0.0029  1655.2705  1.0012
        lockdown[9]   -0.0018   0.0992    0.0016  0.0015  3282.0295  0.9999
       lockdown[10]   -0.0010   0.1272    0.0020  0.0021  3686.0479  0.9998
       lockdown[11]    0.0078   0.1006    0.0016  0.0015  3682.6438  0.9996
       lockdown[12]    0.0038   0.1192    0.0019  0.0022  3075.1532  0.9999
       lockdown[13]   -0.0012   0.1068    0.0017  0.0022  3103.5187  1.0022
       lockdown[14]    0.0455   0.1113    0.0018  0.0024  2079.7790  1.0017
               y[1]   46.4670  23.1671    0.3663  0.3993  3051.0661  0.9997
               y[2]   38.6229  13.0809    0.2068  0.2614  2396.3508  0.9998
               y[3]   19.1768   9.4607    0.1496  0.2225  1618.4441  1.0019
               y[4]   73.7456  33.8283    0.5349  1.0255  1217.8814  1.0047
               y[5]   42.5408  17.5320    0.2772  0.3567  1816.6638  1.0023
               y[6]    9.1561   4.2267    0.0668  0.1047  1837.3744  1.0000
               y[7]   29.0679  17.4434    0.2758  0.2492  3195.2010  0.9993
               y[8]   16.4302  10.7746    0.1704  0.2680  1734.9104  1.0005
               y[9]   71.7330  33.2071    0.5251  0.6524  2360.5565  1.0006
              y[10]  140.8989  52.9734    0.8376  1.2555  1807.0170  1.0019
              y[11]   26.0038  13.2449    0.2094  0.2405  2349.8714  0.9997
              y[12]   82.8426  43.6475    0.6901  0.8820  2657.5728  1.0006
              y[13]   53.4443  25.2102    0.3986  0.4923  2491.2797  0.9997
              y[14]   81.7510  35.6084    0.5630  0.9667  1599.3832  1.0005
          α_hier[1]    0.0250   0.0519    0.0008  0.0009  3402.8745  1.0003
          α_hier[2]    0.0307   0.0571    0.0009  0.0011  2997.5497  1.0022
          α_hier[3]    0.4780   0.1807    0.0029  0.0084   292.5909  1.0134
          α_hier[4]    0.0167   0.0355    0.0006  0.0006  2836.6323  1.0003
          α_hier[5]    1.1276   0.1476    0.0023  0.0045  1166.6317  1.0024
          α_hier[6]    0.0776   0.1171    0.0019  0.0045   447.5792  1.0111
                  γ    0.1047   0.0666    0.0011  0.0031   514.1243  1.0154
                  κ    1.1469   0.2330    0.0037  0.0048  1953.9740  1.0004
               μ[1]    3.9794   0.4695    0.0074  0.0092  2078.0005  0.9996
               μ[2]    3.6130   0.1697    0.0027  0.0039  1763.3903  1.0002
               μ[3]    4.3002   0.4351    0.0069  0.0158   589.1497  1.0091
               μ[4]    4.5535   0.3895    0.0062  0.0130   824.9565  1.0066
               μ[5]    3.9472   0.3079    0.0049  0.0094   829.6550  1.0073
               μ[6]    4.9544   0.3259    0.0052  0.0089  1325.4430  1.0008
               μ[7]    3.8631   0.5352    0.0085  0.0114  1943.5804  1.0003
               μ[8]    6.3826   0.7272    0.0115  0.0213  1225.8573  1.0003
               μ[9]    4.3782   0.5184    0.0082  0.0118  1846.1891  0.9997
              μ[10]    2.4349   0.2378    0.0038  0.0070   931.9334  1.0053
              μ[11]    3.8172   0.3753    0.0059  0.0099  1458.7896  1.0011
              μ[12]    2.0997   0.3078    0.0049  0.0078  1545.5404  1.0019
              μ[13]    3.9692   0.4579    0.0072  0.0119  1471.6664  1.0000
              μ[14]    3.7025   0.3535    0.0056  0.0112  1150.7984  1.0006
                  τ   53.0052  17.4602    0.2761  0.3973  1613.3580  1.0005
                  ϕ    6.8730   0.5511    0.0087  0.0089  3726.1288  1.0002
    
    Quantiles
         parameters     2.5%     25.0%     50.0%     75.0%     97.5%
      ─────────────  ───────  ────────  ────────  ────────  ────────
       ifr_noise[1]   0.8033    0.9298    0.9989    1.0688    1.1966
       ifr_noise[2]   0.8116    0.9329    1.0000    1.0656    1.1942
       ifr_noise[3]   0.8007    0.9265    0.9955    1.0611    1.1846
       ifr_noise[4]   0.8028    0.9335    1.0026    1.0698    1.1942
       ifr_noise[5]   0.7999    0.9292    0.9974    1.0659    1.1977
       ifr_noise[6]   0.8074    0.9268    0.9895    1.0580    1.1852
       ifr_noise[7]   0.7950    0.9274    0.9921    1.0589    1.1930
       ifr_noise[8]   0.7890    0.9226    0.9914    1.0563    1.1945
       ifr_noise[9]   0.8106    0.9353    1.0015    1.0666    1.1948
      ifr_noise[10]   0.8290    0.9479    1.0148    1.0791    1.2145
      ifr_noise[11]   0.7956    0.9296    0.9975    1.0649    1.1865
      ifr_noise[12]   0.8111    0.9339    1.0009    1.0706    1.2025
      ifr_noise[13]   0.8095    0.9360    1.0020    1.0699    1.1932
      ifr_noise[14]   0.8156    0.9396    1.0055    1.0715    1.1985
        lockdown[1]  -0.2208   -0.0530   -0.0022    0.0481    0.2256
        lockdown[2]  -0.2049   -0.0622   -0.0104    0.0249    0.1430
        lockdown[3]  -0.2412   -0.0645   -0.0087    0.0320    0.1893
        lockdown[4]  -0.0408    0.0218    0.0927    0.1907    0.4060
        lockdown[5]  -0.2022   -0.0411    0.0042    0.0572    0.2512
        lockdown[6]  -0.2405   -0.0711   -0.0158    0.0203    0.1414
        lockdown[7]  -0.2870   -0.0666   -0.0074    0.0361    0.2043
        lockdown[8]  -0.3473   -0.1313   -0.0471   -0.0005    0.1028
        lockdown[9]  -0.2196   -0.0470   -0.0004    0.0467    0.2105
       lockdown[10]  -0.2818   -0.0537   -0.0004    0.0509    0.2774
       lockdown[11]  -0.2034   -0.0401    0.0027    0.0551    0.2276
       lockdown[12]  -0.2539   -0.0480    0.0010    0.0562    0.2582
       lockdown[13]  -0.2182   -0.0528   -0.0010    0.0491    0.2289
       lockdown[14]  -0.1552   -0.0158    0.0228    0.0969    0.3229
               y[1]  14.9179   30.0247   41.9126   58.3610  101.7665
               y[2]  19.5299   29.0981   36.6782   45.5189   70.1557
               y[3]   6.2077   12.4272   17.2003   24.0866   42.8143
               y[4]  23.8405   49.3363   68.8392   92.3015  153.7364
               y[5]  15.3591   30.0828   40.2003   52.3791   83.7325
               y[6]   3.4499    6.2449    8.2618   11.1259   19.4929
               y[7]   7.9239   16.9180   24.7996   36.2075   75.5797
               y[8]   4.3036    9.2674   13.7824   20.2839   46.1405
               y[9]  25.7094   47.7998   65.8459   88.0414  153.9687
              y[10]  58.3771  103.2094  134.1771  171.8739  264.0741
              y[11]   8.7082   16.5333   23.4436   32.4592   58.9983
              y[12]  23.2083   51.9796   74.6536  104.0517  191.2370
              y[13]  18.2376   35.3789   49.1850   65.7620  116.0045
              y[14]  30.5740   55.7063   75.7009  101.0443  167.4224
          α_hier[1]   0.0000    0.0000    0.0019    0.0242    0.1824
          α_hier[2]   0.0000    0.0000    0.0036    0.0332    0.2080
          α_hier[3]   0.0026    0.3730    0.4905    0.5978    0.7993
          α_hier[4]   0.0000    0.0000    0.0012    0.0144    0.1276
          α_hier[5]   0.8306    1.0298    1.1296    1.2270    1.4153
          α_hier[6]   0.0000    0.0003    0.0150    0.1156    0.4115
                  γ   0.0081    0.0523    0.0975    0.1456    0.2550
                  κ   0.7561    0.9801    1.1271    1.2899    1.6530
               μ[1]   3.1409    3.6544    3.9459    4.2684    4.9864
               μ[2]   3.2826    3.4992    3.6051    3.7255    3.9561
               μ[3]   3.5900    3.9988    4.2449    4.5507    5.3007
               μ[4]   3.9220    4.2754    4.5081    4.7815    5.4161
               μ[5]   3.4703    3.7330    3.8936    4.1225    4.6746
               μ[6]   4.3393    4.7278    4.9528    5.1696    5.6089
               μ[7]   2.9072    3.4929    3.8364    4.1970    5.0471
               μ[8]   4.9653    5.8929    6.3766    6.8616    7.8508
               μ[9]   3.4333    4.0293    4.3408    4.6990    5.5073
              μ[10]   2.0695    2.2673    2.3974    2.5695    2.9983
              μ[11]   3.1598    3.5554    3.7920    4.0572    4.5967
              μ[12]   1.5810    1.8869    2.0716    2.2842    2.7769
              μ[13]   3.1791    3.6543    3.9293    4.2499    4.9921
              μ[14]   3.0808    3.4508    3.6740    3.9262    4.4323
                  τ  26.6958   40.4641   50.7268   62.4155   94.9594
                  ϕ   5.8429    6.5022    6.8514    7.2332    8.0291

```julia
plot(chains_posterior[[:κ, :ϕ, :τ]]; α = .5, linewidth=1.5)
```

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/uk-posterior-kappa-phi-tau-sample-plot.png)

```julia
# Compute generated quantities for the chains pooled together
pooled_chains = MCMCChains.pool_chain(chains_posterior)
generated_posterior = vectup2tupvec(generated_quantities(m, pooled_chains));

daily_cases_posterior, daily_deaths_posterior, Rt_posterior, Rt_adj_posterior = generated_posterior;
```

The posterior predictive distribution:

```julia
country_prediction_plot(uk_index, daily_cases_posterior, daily_deaths_posterior, Rt_posterior; main_title = "(posterior)")
```

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/uk-predictive-posterior-Rt.png)

and with the adjusted \\(R\_t\\):

```julia
country_prediction_plot(uk_index, daily_cases_posterior, daily_deaths_posterior, Rt_adj_posterior; main_title = "(posterior)")
```

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/uk-predictive-posterior-Rt-adjusted.png)


## All countries: prior vs. posterior predictive

For the sake of completeness, here are the prior and posterior predictive distributions for all the 14 countries in a side-by-side comparison.

{% wrap two-by-two %}

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-prior-predictive-01.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-posterior-predictive-01.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-prior-predictive-02.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-posterior-predictive-02.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-prior-predictive-03.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-posterior-predictive-03.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-prior-predictive-04.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-posterior-predictive-04.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-prior-predictive-05.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-posterior-predictive-05.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-prior-predictive-06.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-posterior-predictive-06.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-prior-predictive-07.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-posterior-predictive-07.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-prior-predictive-08.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-posterior-predictive-08.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-prior-predictive-09.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-posterior-predictive-09.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-prior-predictive-10.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-posterior-predictive-10.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-prior-predictive-11.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-posterior-predictive-11.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-prior-predictive-12.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-posterior-predictive-12.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-prior-predictive-13.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-posterior-predictive-13.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-prior-predictive-14.png)

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/country-posterior-predictive-14.png)

{% endwrap %}


## What if we didn't do any/certain interventions?

One interesting thing one can do after obtaining estimates for the effect of each of the interventions is to run the model but now *without* all or a subset of the interventions performed. Thus allowing us to get a sense of what the outcome would have been without those interventions, and also whether or not the interventions have the wanted effect.

`data.covariates[m]` is a binary matrix for each `m` (country index), with `data.covariate[m][:, k]` then being a binary vector representing the time-series for the k-th covariate: `0` means the intervention has is not implemented, `1` means that the intervention is implemented. As an example, if schools and universites were closed after the 45th day for country `m`, then `data.covariate[m][1:45, k]` are all zeros and `data.covariate[m][45:end, k]` are all ones.

```julia
# Get the index of schools and univerities closing
schools_universities_closed_index = findfirst(==("schools_universities"), names_covariates)
# Time-series for UK
data.covariates[uk_index][:, schools_universities_closed_index]
```

    100-element Array{Float64,1}:
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     ⋮
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0

Notice that the above assumes that not only are schools and universities closed *at some point*, but rather that they also stay closed in the future (at the least the future that we are considering).

Therefore we can for example simulate "what happens if we never closed schools and universities?" by instead setting this entire vector to `0` and re-run the model on the infererred parameters, similar to what we did before to compute the "generated quantities", e.g. \\(R\_t\\).

{% details Convenience function for zeroing out subsets of the interventions %}

```julia
"""
    zero_covariates(xs::AbstractMatrix{<:Real}; remove=[], keep=[])

Allows you to zero out covariates if the name of the covariate is in `remove` or NOT zero out those in `keep`.
Note that only `remove` xor `keep` can be non-empty.

Useful when instantiating counter-factual models, as it allows one to remove/keep a subset of the covariates.
"""
zero_covariates(xs::AbstractMatrix{<:Real}; kwargs...) = zero_covariates(xs, names_covariates; kwargs...)
function zero_covariates(xs::AbstractMatrix{<:Real}, names_covariates; remove=[], keep=[])
    @assert (isempty(remove) || isempty(keep)) "only `remove` or `keep` can be non-empty"

    if isempty(keep)
        return mapreduce(hcat, enumerate(eachcol(xs))) do (i, c)
            (names_covariates[i] ∈ remove ? zeros(eltype(c), length(c)) : c)
        end
    else
        return mapreduce(hcat, enumerate(eachcol(xs))) do (i, c)
            (names_covariates[i] ∈ keep ? c : zeros(eltype(c), length(c))) 
        end
    end
end
```

    zero_covariates (generic function with 2 methods)

{% enddetails %}

Now we can consider simulation under the posterior with *no* intervention, and we're going to visualize the respective *portions* of the population by rescaling by total population:

```julia
# What happens if we don't do anything?
m_counterfactual = model_def(
    data.num_impute,
    data.num_total_days,
    data.cases,
    data.deaths,
    data.π,
    [zeros(size(c)) for c in data.covariates], # <= remove ALL covariates
    data.epidemic_start,
    data.population,
    data.serial_intervals,
    lockdown_index,
    true # <= use full model
);

# Compute the "generated quantities" for the "counter-factual" model
generated_counterfactual = vectup2tupvec(generated_quantities(m_counterfactual, pooled_chains));
daily_cases_counterfactual, daily_deaths_counterfactual, Rt_counterfactual, Rt_adj_counterfactual = generated_counterfactual;
country_prediction_plot(5, daily_cases_counterfactual, daily_deaths_counterfactual, Rt_adj_counterfactual; normalize_pop = true)
```

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/counterfactual-remove-all.png)

We can also consider the cases where we only do *some* of the interventions, e.g. we never do a full lockdown (`lockdown`) or close schools and universities (`schools_universities`):

```julia
# What happens if we never close schools nor do a lockdown?
m_counterfactual = model_def(
    data.num_impute,
    data.num_total_days,
    data.cases,
    data.deaths,
    data.π,
    [zero_covariates(c; remove = ["lockdown", "schools_universities"]) for c in data.covariates], # <= remove covariates
    data.epidemic_start,
    data.population,
    data.serial_intervals,
    lockdown_index,
    true # <= use full model
);

# Compute the "generated quantities" for the "counter-factual" model
generated_counterfactual = vectup2tupvec(generated_quantities(m_counterfactual, pooled_chains));
daily_cases_counterfactual, daily_deaths_counterfactual, Rt_counterfactual, Rt_adj_counterfactual = generated_counterfactual;
country_prediction_plot(uk_index, daily_cases_counterfactual, daily_deaths_counterfactual, Rt_adj_counterfactual; normalize_pop = true)
```

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/counterfactual-remove-lockdown-and-schools.png)

As mentioned, this assumes that we will stay in lockdown and schools and universities will be closed in the future. We can also consider, say, removing the lockdown, i.e. opening up, at some future point in time:

```julia
lift_lockdown_time = 75

new_covariates = [copy(c) for c in data.covariates] # <= going to do inplace manipulations so we copy
for covariates_m ∈ new_covariates
    covariates_m[lift_lockdown_time:end, lockdown_index] .= 0
end

# What happens if we never close schools nor do a lockdown?
m_counterfactual = model_def(
    data.num_impute,
    data.num_total_days,
    data.cases,
    data.deaths,
    data.π,
    new_covariates,
    data.epidemic_start,
    data.population,
    data.serial_intervals,
    lockdown_index,
    true # <= use full model
);

# Compute the "generated quantities" for the "counter-factual" model
generated_counterfactual = vectup2tupvec(generated_quantities(m_counterfactual, pooled_chains));
daily_cases_counterfactual, daily_deaths_counterfactual, Rt_counterfactual, Rt_adj_counterfactual = generated_counterfactual;
country_prediction_plot(uk_index, daily_cases_counterfactual, daily_deaths_counterfactual, Rt_adj_counterfactual; normalize_pop = true)
```

![nil](../assets/figures/2020-05-04-Imperial-Report13-analysis/counterfactual-remove-lockdown-after-a-while.png)


# Conclusion

Well, there isn't one. As stated before, drawing conclusions is not the purpose of this document. With that being said, we *are* working on exploring this and other models further, e.g. relaxing certain assumptions, model validation & comparison, but this will hopefully be available in a more technical and formal report sometime in the near future after proper validation and analysis. But since time is of the essence in these situations, we thought it was important to make the above and related code available to the public immediately. At the very least it should be comforting to know that two different PPLs both produce the same inference results when the model might be used to inform policy decisions on a national level.

If you have any questions or comments, feel free to reach out either on the [Github repo](https://github.com/TuringLang/Covid19) or to any of us personally.


<!----- Footnotes ----->

[^fn1]: The issue with *not* having the `max` is that it's possible to obtain a negative \\(R\_t\\) which is bad for two reasons: 1) it doesn't make sense with negative spreading of the virus, 2) it leads to invalid parameterization for `NegativeBinomial2`. In Stan a invalid parameterization will be considered a rejected sample, and thus these samples will be rejected. In the case of `Turing.jl`, if \\(R\_t = 0\\), then observing a *positive* number for daily deaths will result in `-Inf` added to the log-pdf and so the sample will also be rejected. Hence, both PPLs will arrive at "correct" inference but with different processes.
[^fn2]: You *could* use something like [Atomic in Julia](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.Atomic), but it comes at a unnecessary performance overhead in this case.
[^fn3]: Recently one of our team-members joined as a maintainer of [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl) to make sure that `Turing.jl` also has fast and reliable reverse-mode differentiation. ReverseDiff.jl is already compatible with `Turing.jl`, but we hope that this will help make if much, much faster.
