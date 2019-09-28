# The Sampling Interface

Turing implements a sampling interface that is intended to provide a common framework for Markov Chain Monte Carlo samplers. The interface presents several structures and functions that one needs to overload in order to implement an interface-compatible sampler. 

This guide will demonstrate how to implement the interface without Turing, and then demonstrate how to implement the same model with Turing.

## Interface Overview

Any implementation of an inference method that uses Turing's MCMC interface should implement some combination of the following types and functions:

1. A subtype of `AbstractSampler`, defined as a mutable struct containing state information or sampler parameters.
2. A subtype of `AbstractTransition`, which represents a single draw from the sampler.
3. A function `transition_type` which returns the `AbstractTransition` type used by an implementation of an `AbstractSampler`, or a function `transition_init` with returns a `Vector{AbstractTransition}` of length `N`.
4. A function `sample_init!` which performs any necessary set up. 
5. A function `step!` which returns an `AbstractTransition`.
6. A function `sample_end!` which handles any sampler wrap-up.
7. A function `MCMCChains.Chains` which accepts an `Vector{<:AbstractTransition}` and returns an `MCMCChains` object.

The interface methods with exclamation points are those that are intended to allow for some state mutation. Any mutating function is meant to allow mutation where needed -- you might use:

- `sample_init!` to run some kind of sampler preparation, before sampling begins. This could mutate a sampler's state.
- `step!` might mutate a sampler flag after each sample. MH does this for example by using a `violating_support` flag.
- `sample_end!` contains any wrap-up you might need to do. If you were sampling in a transformed space, this might be where you convert everything back to a constrained space.

## Implementing Metropolis-Hastings without Turing

[Metropolis-Hastings](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) is often the first sampling method that people are exposed to. It is a very straightforward algorithm and is accordingly the easiest to implement, so it makes for a good example. In this section, you will learn how to use the types and functions listed above to implement the Metropolis-Hastings sampler using the MCMC interface.

### Imports

Let's begin by importing the relevant libraries. We'll import `Turing.Interface`, which contains the interface framework we'll fill out. We also need `Distributions` and `Random`.

```julia
# Import the relevant libraries.
using Turing.Interface
using Distributions
using Random
```

An interface extension (like the one we're writing right now) typically requires that you overload or implement several functions. Specifically, you should `import` the functions you intend to overload. This next code block accomplishes that. 

From `Distributions`, we need `Sampleable`, `VariateForm`, and `ValueSupport`, three abstract types that define a distribution. Models in the interface are assumed to be subtypes of `Sampleable{VariateForm, ValueSupport}`. In this section our model is going be be extremely simple, so we will not end up using these except to make sure that the inference functions are dispatching correctly.

We need the `MCMCChains.Chains` function, because at the end of sampling we need to be able to convert a vector of `<:AbstractTransitions` to an `MCMCChains` object. Currently, the plan is to eventually move all the interface methods from `Turing.Interface` to `MCMCChains.Interface`, which is why this step needs to be done for the moment.

Lastly, we import the specific functions from the `Interface` that we need to overload. Because Metropolis-Hastings is so simple we only need a few functions/types:

- `step!`
- `AbstractSampler`
- `AbstractTransition`
- `transition_type`

More complex inference methods may need to import additional functions. For now, let's focus on the code:

```julia
# Import specific functions and types to use or overload.
import Distributions: VariateForm, ValueSupport, variate_form, value_support, Sampleable
import MCMCChains: Chains
import Turing.Interface: step!, AbstractSampler, AbstractTransition, transition_type
```

### Sampler

Let's begin our sampler definition by defining a sampler called `MetropolisHastings` which is a subtype of `AbstractSampler`. Correct typing is very important for proper interface implementation -- if you are missing a subtype, your inference method will likely not work when you call `sample`.

```julia
# Define a sampler type.
struct MetropolisHastings{T} <: AbstractSampler 
    init_θ :: T
end
```

Above, we have defined a sampler that only stores the initial parameterization of the prior. You can have a struct that has no fields, and simply use it for dispatching onto the relevant functions, or you can store a large amount of state information in your sampler. This implementation of Metropolis-Hastings is a pure immutable one, and no state information need be saved. 

The general intuition for what to store in your sampler struct is that anything you may need to perform inference between samples but you don't want to store in an `AbstractTransition` should go into the sampler struct. It's the only way you can carry non-sample related state information between `step!` calls. 

### Model

Next, we need to have a model of some kind. A model is a struct that's a subtype of `Sampleable{VariateForm, ValueSupport}` that contains whatever information is necessary to perform inference on your problem. In our case we want to know the mean and variance parameters for a standard Normal distribution, so we can keep our model to the log density of a Normal. 

Note that we only have to do this because we are not yet integrating the sampler with Turing -- Turing has a very sophisticated modelling engine that removes the need to define custom model structs. 

```julia
# Define a model type. Stores the log density function and the data to 
# evaluate the log density on.
struct DensityModel{V<:VariateForm, S<:ValueSupport, T} <: Sampleable{V, S}
    π :: Function
    data :: T
end

# Default density constructor.
DensityModel(π::Function, data::T) where T = DensityModel{VariateForm, ValueSupport, T}(π, data)
```

The `DensityModel` struct has two fields:

1. `π`, which stores some log-density function. 
2. `data`, which stores the data to evaluate the model on.

You could combine these two by making data be part of the `π` function, but for now we're mostly concerned with getting something that works well.

We may also want a helper function that calculates the log probability of a model and a set of parameters. The two functions below provide the functionality we need. The first signature accepts any `θ`, whether it's a vector or a single value, and the the second signature unpacks a `Transition` type to give us a little bit of syntactic sugar. We'll explain what the `Transition` looks like in the following section.

```julia
# Calculate the density of the model given some parameterization.
ℓπ(model::DensityModel, θ::T) where T = model.π(model.data, θ)
ℓπ(model::DensityModel, t::Transition) = t.lp
```

### Transition

The next step is to define some subtype of `AbstractTransition` which we will return from each `step!` call. We'll keep it simple by just defining a wrapper struct that contains the parameter draws and the log density of that draw:

```julia
# Create a very basic Transition type, only stores the 
# parameter draws and the log probability of the draw.
struct Transition{T} <: AbstractTransition
    θ :: T
    lp :: Float64
end

# A helpful constructor to store the new draw and its log density.
Transition(model::DensityModel, θ::T) where T = Transition(θ, ℓπ(model, θ))
```

`Transition` can now store any type of parameter, whether it's a vector of draws from multiple parameters or a single univariate draw. We should also tell the interface what specific subtype of `AbstractTransition` we're going to use, so we can just define a new method on `transition_type`:

```julia
# Tell the interface what transition type we would like to use.
transition_type(spl::MetropolisHastings) = Transition
```

This method only returns the abstract type, `Transition`, which can cause type instability. We can actually be a little more precise here than returning the abstract type. We can concretely type all the `Transition`s we expect to return, because the sampler object contains the initial parameterization:

```julia
# Tell the interface exactly what our type is,
# by constructing a new interface and retrieving
# it's full parameterization.
transition_type(model::DensityModel, spl::MetropolisHastings) = typeof(Transition(spl.init_θ, ℓπ(model, spl.init_θ)))
```

### Metropolis-Hastings

Now it's time to get into the actual inference method. We've defined all of the core pieces we need, but we need to implement the `step!` functions which actually perform our inference.

As a refresher, Metropolis-Hastings implements a very basic algorithm:

1. Pick some initial state, $\theta_0$.
2. For $t$ in $[1,N]$, do
    
    a. Generate a proposal parameterization $θ'_t \sim q(\theta'_t \mid \theta_{t-1})$. 

    b. Calculate the acceptance probability, $\alpha = \text{min}\Big[1,\frac{\pi(θ'_t)}{\pi(\theta_{t-1})} \frac{q(θ_{t-1} \mid θ'_t)}{q(θ'_t \mid θ_{t-1})}) \Big]$.

    c. If $U \le α$ where $U \sim [0,1]$, then $\theta_t = \theta'_t$. Otherwise, $\theta_t = \theta_{t-1}$.

Of course, it's much easier to do this in the log space, so the acceptance probability is more commonly written as 

$$
\alpha = \min\Big[\log \pi(θ'_t) - \log \pi(θ_{t-1}) + \log q(θ_{t-1} \mid θ'_t) - \log q(θ'_t \mid θ_{t-1}), 0\Big]
$$

In interface terms, we should do the following:

1. Make a new transition containing a proposed sample.
2. Calculate the acceptance probability.
3. If we accept, return the new transition, otherwise, return the old one.

### Steps

The `step!` function is the function that performs the bulk of your inference. In our case, we will implement two `step!` functions -- one for the very first iteration, and one for every subsequent iteration.

```julia
# Define the first step! function, which is called at the 
# beginning of sampling. Return the initial parameter used
# to define the sampler.
function step!(
    rng::AbstractRNG,
    model::M,
    spl::S,
    N::Integer;
    kwargs...
) where {
    M <: DensityModel,
    S <: MetropolisHastings
}
    return Transition(model, spl.init_θ)
end
```

The first `step!` function just packages up the initial parameterization inside the sampler, and returns it. We implicity accept the very first parameterization.

The other `step!` function performs the usual steps from Metropolis-Hastings. Included are several helper functions, `proposal` and `q`, which are designed to replicate the functions in the pseudocode above.

- `proposal` generates a new proposal in the form of a `Transition`, which can be univariate if the value passed in is univariate, or it can be multivariate if the `Transition` given is multivariate. Proposals use a basic `Normal` or `MvNormal` proposal distribution.
- `q` returns the log density of one parameterization conditional on another, according to the proposal distribution.
- `step!` generates a new proposal, checks the acceptance probability, and then returns either the previous transition or the proposed transition.

```julia
# Define a function that makes a basic proposal depending on a univariate
# parameterization or a multivariate parameterization.
proposal(spl::MetropolisHastings, model::DensityModel, θ::Real) = Transition(model, rand(Normal(θ, 1)))
proposal(spl::MetropolisHastings, model::DensityModel, θ::Vector{<:Real}) = Transition(model, rand(MvNormal(θ, 1)))
proposal(spl::MetropolisHastings, model::DensityModel, t::Transition) = proposal(spl, model, t.θ)

# Calculate the logpdf of one proposal given another proposal.
q(spl::MetropolisHastings, θ1::Real, θ2::Real) = logpdf(Normal(θ1, 1.0), θ2)
q(spl::MetropolisHastings, θ1::Vector{<:Real}, θ2::Vector{<:Real}) = logpdf(MvNormal(θ1, 1.0), θ2)
q(spl::MetropolisHastings, t1::Transition, t2::Transition) = q(spl, t1.θ, t2.θ)

# Define the other step function. Returns a Transition containing
# either a new proposal (if accepted) or the previous proposal 
# (if not accepted).
function step!(
    rng::AbstractRNG,
    model::M,
    spl::S,
    ::Integer,
    θ_prev::T;
    kwargs...
) where {
    M <: DensityModel,
    S <: MetropolisHastings,
    T <: Transition
}
    # Generate a new proposal.
    θ = proposal(spl, model, θ_prev)
    
    # Calculate the log acceptance probability.
    α = ℓπ(model, θ) - ℓπ(model, θ_prev) + q(spl, θ_prev, θ) - q(spl, θ, θ_prev)

    # Decide whether to return the previous θ or the new one.
    if log(rand()) < min(α, 0.0)
        return θ
    else
        return θ_prev
    end
end
```

### Chains

The last piece in our puzzle is a `Chains` function, which accepts a `Vector{T<:Transition}` and returns an `MCMCChains.Chains` struct. Fortunately, our `Transition` is incredibly simple, and we only need to build a little bit of functionality to accept custom parameter names passed in by the user.

```julia
# A basic chains constructor that works with the Transition struct we defined.
function Chains(
    rng::AbstractRNG, 
    ℓ::DensityModel, 
    s::MetropolisHastings, 
    N::Integer, 
    ts::Vector{T}; 
    param_names=missing,
    kwargs...
) where {T <: Transition}
    # Turn all the transitions into a vector-of-vectors.
    vals = [vcat(t.θ, t.lp) for t in ts]

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

That was the final piece!

### Testing the implementation

Now that we have all the pieces, we should test the implementation by defining a model to calculate the mean and variance parameters of a Normal distribution. We can do this by constructing a target density function, providing a sample of data, and then running the sampler with `sample`.

```julia
# Define the components of a basic model.
insupport(θ) = θ[2] >= 0
dist(θ) = Normal(θ[1], θ[2])
density(data, θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf

# Generate a set of data from the posterior we want to estimate.
data = rand(Normal(5,3), 10)

# Construct a DensityModel.
model = DensityModel(density, data)

# Set up our sampler.
spl = MetropolisHastings([0.0, 0.0])

# Sample from the posterior.
chain = sample(model, spl, 10000; param_names=["μ", "σ"])
```

If all the interface functions have been extended properly, you should get an output from `display(chain)` that looks something like this:

```
Object of type Chains, with data of type 10000×3×1 Array{Float64,3}

Iterations        = 1:10000
Thinning interval = 1
Chains            = 1
Samples per chain = 10000
internals         = lp
parameters        = μ, σ

2-element Array{MCMCChains.ChainDataFrame,1}

Summary Statistics
. Omitted printing of 1 columns
│ Row │ parameters │ mean    │ std      │ naive_se   │ mcse      │ ess     │
│     │ Symbol     │ Float64 │ Float64  │ Float64    │ Float64   │ Any     │
├─────┼────────────┼─────────┼──────────┼────────────┼───────────┼─────────┤
│ 1   │ μ          │ 5.21391 │ 1.09406  │ 0.0109406  │ 0.0342099 │ 900.188 │
│ 2   │ σ          │ 3.47312 │ 0.934172 │ 0.00934172 │ 0.0307114 │ 905.218 │

Quantiles

│ Row │ parameters │ 2.5%    │ 25.0%   │ 50.0%   │ 75.0%   │ 97.5%   │
│     │ Symbol     │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │
├─────┼────────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ 1   │ μ          │ 2.98559 │ 4.53086 │ 5.21764 │ 5.93417 │ 7.33767 │
│ 2   │ σ          │ 2.14871 │ 2.8051  │ 3.29586 │ 3.96179 │ 5.75031 │
```

It looks like we're extremely close to our true parameters of `Normal(5,3)`, though with a fairly high variance due to the low sample size.

