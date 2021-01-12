---
title: Interface Guide
toc: true
---

# The sampling interface

Turing implements a sampling interface (hosted at
[AbstractMCMC](https://github.com/TuringLang/AbstractMCMC.jl)) that is intended to provide
a common framework for Markov chain Monte Carlo samplers. The interface presents several
structures and functions that one needs to overload in order to implement an
interface-compatible sampler. 

This guide will demonstrate how to implement the interface without Turing.

## Interface overview

Any implementation of an inference method that uses the AbstractMCMC interface should
implement a subset of the following types and functions:

1. A subtype of `AbstractSampler`, defined as a mutable struct containing state information or sampler parameters.
2. A function `sample_init!` which performs any necessary set-up (default: do not perform any set-up). 
3. A function `step!` which returns a transition that represents a single draw from the sampler.
4. A function `transitions_init` which returns a container for the transitions obtained from the sampler
   (default: return a `Vector{T}` of length `N` where `T` is the type of the transition obtained in the first step and `N` is the number of requested samples).
5. A function `transitions_save!` which saves transitions to the container (default: save the transition of iteration `i`
   at position `i` in the vector of transitions).
6. A function `sample_end!` which handles any sampler wrap-up (default: do not perform any wrap-up).
7. A function `bundle_samples` which accepts the container of transitions and returns a collection of samples
   (default: return the vector of transitions).

The interface methods with exclamation points are those that are intended to allow for
state mutation. Any mutating function is meant to allow mutation where needed -- you might
use:

- `sample_init!` to run some kind of sampler preparation, before sampling begins. This
  could mutate a sampler's state.
- `step!` might mutate a sampler flag after each sample. 
- `sample_end!` contains any wrap-up you might need to do. If you were sampling in a
  transformed space, this might be where you convert everything back to a constrained space.

## Why do you have an interface?

The motivation for the interface is to allow Julia's fantastic probabilistic programming
language community to have a set of standards and common implementations so we can all
thrive together. Markov chain Monte Carlo methods tend to have a very similar framework to
one another, and so a common interface should help more great inference methods built in
single-purpose packages to experience more use among the community. 

## Implementing Metropolis-Hastings without Turing

[Metropolis-Hastings](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) is often the
first sampling method that people are exposed to. It is a very straightforward algorithm
and is accordingly the easiest to implement, so it makes for a good example. In this
section, you will learn how to use the types and functions listed above to implement the
Metropolis-Hastings sampler using the MCMC interface.

The full code for this implementation is housed in
[AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl).

### Imports

Let's begin by importing the relevant libraries. We'll import `AbstracMCMC`, which contains
the interface framework we'll fill out. We also need `Distributions` and `Random`.

```julia
# Import the relevant libraries.
import AbstractMCMC
using Distributions
using Random
```

An interface extension (like the one we're writing right now) typically requires that you overload or implement several functions. Specifically, you should `import` the functions you intend to overload. This next code block accomplishes that. 

From `Distributions`, we need `Sampleable`, `VariateForm`, and `ValueSupport`, three abstract types that define a distribution. Models in the interface are assumed to be subtypes of `Sampleable{VariateForm, ValueSupport}`. In this section our model is going be be extremely simple, so we will not end up using these except to make sure that the inference functions are dispatching correctly.

### Sampler

Let's begin our sampler definition by defining a sampler called `MetropolisHastings` which
is a subtype of `AbstractSampler`. Correct typing is very important for proper interface
implementation -- if you are missing a subtype, your method may not be dispatched to when
you call `sample`.

```julia
# Define a sampler type.
struct MetropolisHastings{T, D} <: AbstractMCMC.AbstractSampler 
    init_θ::T
    proposal::D
end

# Default constructors.
MetropolisHastings(init_θ::Real) = MetropolisHastings(init_θ, Normal(0,1))
MetropolisHastings(init_θ::Vector{<:Real}) = MetropolisHastings(init_θ, MvNormal(length(init_θ),1))
```

Above, we have defined a sampler that stores the initial parameterization of the prior,
and a distribution object from which proposals are drawn. You can have a struct that has no
fields, and simply use it for dispatching onto the relevant functions, or you can store a
large amount of state information in your sampler. 

The general intuition for what to store in your sampler struct is that anything you may
need to perform inference between samples but you don't want to store in a transition
should go into the sampler struct. It's the only way you can carry non-sample related state
information between `step!` calls.

### Model

Next, we need to have a model of some kind. A model is a struct that's a subtype of
`AbstractModel` that contains whatever information is necessary to perform inference on
your problem. In our case we want to know the mean and variance parameters for a standard
Normal distribution, so we can keep our model to the log density of a Normal. 

Note that we only have to do this because we are not yet integrating the sampler with Turing
-- Turing has a very sophisticated modelling engine that removes the need to define custom
model structs. 

```julia
# Define a model type. Stores the log density function.
struct DensityModel{F<:Function} <: AbstractMCMC.AbstractModel
    ℓπ::F
end
```

### Transition

The next step is to define some transition which we will return from each `step!` call.
We'll keep it simple by just defining a wrapper struct that contains the parameter draws
and the log density of that draw:

```julia
# Create a very basic Transition type, only stores the 
# parameter draws and the log probability of the draw.
struct Transition{T, L}
    θ::T
    lp::L
end

# Store the new draw and its log density.
Transition(model::DensityModel, θ) = Transition(θ, ℓπ(model, θ))
```

`Transition` can now store any type of parameter, whether it's a vector of draws from
multiple parameters or a single univariate draw.

### Metropolis-Hastings

Now it's time to get into the actual inference. We've defined all of the core pieces we
need, but we need to implement the `step!` function which actually performs inference.

As a refresher, Metropolis-Hastings implements a very basic algorithm:

1. Pick some initial state, \$\$\theta\_0\$\$.
2. For \$\$t\$\$ in \$\$[1,N]\$\$, do
    
    a. Generate a proposal parameterization \$\$θ'\_t \sim q(\theta'\_t \mid \theta\_{t-1})\$\$. 

    b. Calculate the acceptance probability, \$\$\alpha = \text{min}\Big[1,\frac{\pi(θ'\_t)}{\pi(\theta\_{t-1})} \frac{q(θ\_{t-1} \mid θ'\_t)}{q(θ'\_t \mid θ\_{t-1})}) \Big]\$\$.

    c. If \$\$U \le α\$\$ where \$\$U \sim [0,1]\$\$, then \$\$\theta\_t = \theta'\_t\$\$. Otherwise, \$\$\theta\_t = \theta\_{t-1}\$\$.

Of course, it's much easier to do this in the log space, so the acceptance probability is
more commonly written as 

\$\$\log \alpha = \min\Big[0, \log \pi(θ'\_t) - \log \pi(θ\_{t-1}) + \log q(θ\_{t-1} \mid θ'\_t) - \log q(θ'\_t \mid θ\_{t-1})]\$\$

In interface terms, we should do the following:

1. Make a new transition containing a proposed sample.
2. Calculate the acceptance probability.
3. If we accept, return the new transition, otherwise, return the old one.

### Steps

The `step!` function is the function that performs the bulk of your inference. In our case,
we will implement two `step!` functions -- one for the very first iteration, and one for
every subsequent iteration.

```julia
# Define the first step! function, which is called at the 
# beginning of sampling. Return the initial parameter used
# to define the sampler.
function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::DensityModel,
    spl::MetropolisHastings,
    N::Integer,
    ::Nothing;
    kwargs...
)
    return Transition(model, spl.init_θ)
end
```

The first `step!` function just packages up the initial parameterization inside the
sampler, and returns it. We implicity accept the very first parameterization.

The other `step!` function performs the usual steps from Metropolis-Hastings. Included are
several helper functions, `proposal` and `q`, which are designed to replicate the functions
in the pseudocode above.

- `proposal` generates a new proposal in the form of a `Transition`, which can be
  univariate if the value passed in is univariate, or it can be multivariate if the
  `Transition` given is multivariate. Proposals use a basic `Normal` or `MvNormal` proposal
  distribution.
- `q` returns the log density of one parameterization conditional on another, according to
  the proposal distribution.
- `step!` generates a new proposal, checks the acceptance probability, and then returns
  either the previous transition or the proposed transition.

```julia
# Define a function that makes a basic proposal depending on a univariate
# parameterization or a multivariate parameterization.
propose(spl::MetropolisHastings, model::DensityModel, θ::Real) = 
    Transition(model, θ + rand(spl.proposal))
propose(spl::MetropolisHastings, model::DensityModel, θ::Vector{<:Real}) = 
    Transition(model, θ + rand(spl.proposal))
propose(spl::MetropolisHastings, model::DensityModel, t::Transition) =
    propose(spl, model, t.θ)

# Calculates the probability `q(θ|θcond)`, using the proposal distribution `spl.proposal`.
q(spl::MetropolisHastings, θ::Real, θcond::Real) = logpdf(spl.proposal, θ - θcond)
q(spl::MetropolisHastings, θ::Vector{<:Real}, θcond::Vector{<:Real}) =
    logpdf(spl.proposal, θ - θcond)
q(spl::MetropolisHastings, t1::Transition, t2::Transition) = q(spl, t1.θ, t2.θ)

# Calculate the density of the model given some parameterization.
ℓπ(model::DensityModel, θ) = model.ℓπ(θ)
ℓπ(model::DensityModel, t::Transition) = t.lp

# Define the other step function. Returns a Transition containing
# either a new proposal (if accepted) or the previous proposal 
# (if not accepted).
function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::DensityModel,
    spl::MetropolisHastings,
    ::Integer,
    θ_prev::Transition;
    kwargs...
)
    # Generate a new proposal.
    θ = propose(spl, model, θ_prev)

    # Calculate the log acceptance probability.
    α = ℓπ(model, θ) - ℓπ(model, θ_prev) + q(spl, θ_prev, θ) - q(spl, θ, θ_prev)

    # Decide whether to return the previous θ or the new one.
    if log(rand(rng)) < min(α, 0.0)
        return θ
    else
        return θ_prev
    end
end
```

### Chains

In the default implementation, `sample` just returns a vector of all transitions. If
instead you would like to obtain a `Chains` object (e.g., to simplify downstream analysis),
you have to implement the `bundle_samples` function as well. It accepts the vector of
transitions and returns a collection of samples. Fortunately, our `Transition` is
incredibly simple, and we only need to build a little bit of functionality to accept custom
parameter names passed in by the user.

```julia
# A basic chains constructor that works with the Transition struct we defined.
function AbstractMCMC.bundle_samples(
    rng::AbstractRNG, 
    ℓ::DensityModel, 
    s::MetropolisHastings, 
    N::Integer, 
    ts::Vector{<:Transition},
    chain_type::Type{Any};
    param_names=missing,
    kwargs...
)
    # Turn all the transitions into a vector-of-vectors.
    vals = copy(reduce(hcat,[vcat(t.θ, t.lp) for t in ts])')

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = ["Parameter $i" for i in 1:(length(first(vals))-1)]
    end

    # Add the log density field to the parameter names.
    push!(param_names, "lp")

    # Bundle everything up and return a Chains struct.
    return Chains(vals, param_names, (internals=["lp"],))
end
```

All done!

You can even implement different output formats by implementing `bundle_samples` for
different `chain_type`s, which can be provided as keyword argument to `sample`. As default
`sample` uses `chain_type = Any`.

### Testing the implementation

Now that we have all the pieces, we should test the implementation by defining a model to
calculate the mean and variance parameters of a Normal distribution. We can do this by
constructing a target density function, providing a sample of data, and then running the
sampler with `sample`.

```julia
# Generate a set of data from the posterior we want to estimate.
data = rand(Normal(5, 3), 30)

# Define the components of a basic model.
insupport(θ) = θ[2] >= 0
dist(θ) = Normal(θ[1], θ[2])
density(θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf

# Construct a DensityModel.
model = DensityModel(density)

# Set up our sampler with initial parameters.
spl = MetropolisHastings([0.0, 0.0])

# Sample from the posterior.
chain = sample(model, spl, 100000; param_names=["μ", "σ"])
```

If all the interface functions have been extended properly, you should get an output from
`display(chain)` that looks something like this:

```
Object of type Chains, with data of type 100000×3×1 Array{Float64,3}

Iterations        = 1:100000
Thinning interval = 1
Chains            = 1
Samples per chain = 100000
internals         = lp
parameters        = μ, σ

2-element Array{ChainDataFrame,1}

Summary Statistics

│ Row │ parameters │ mean    │ std      │ naive_se   │ mcse       │ ess     │ r_hat   │
│     │ Symbol     │ Float64 │ Float64  │ Float64    │ Float64    │ Any     │ Any     │
├─────┼────────────┼─────────┼──────────┼────────────┼────────────┼─────────┼─────────┤
│ 1   │ μ          │ 5.33157 │ 0.854193 │ 0.0027012  │ 0.00893069 │ 8344.75 │ 1.00009 │
│ 2   │ σ          │ 4.54992 │ 0.632916 │ 0.00200146 │ 0.00534942 │ 14260.8 │ 1.00005 │

Quantiles

│ Row │ parameters │ 2.5%    │ 25.0%   │ 50.0%   │ 75.0%   │ 97.5%   │
│     │ Symbol     │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │
├─────┼────────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ 1   │ μ          │ 3.6595  │ 4.77754 │ 5.33182 │ 5.89509 │ 6.99651 │
│ 2   │ σ          │ 3.5097  │ 4.09732 │ 4.47805 │ 4.93094 │ 5.96821 │
```

It looks like we're extremely close to our true parameters of `Normal(5,3)`, though with a
fairly high variance due to the low sample size.

## Conclusion

We've seen how to implement the sampling interface for general projects. Turing's interface
methods are ever-evolving, so please open an issue at
[AbstractMCMC](https://github.com/TuringLang/AbstractMCMC.jl) with feature requests or
problems.
