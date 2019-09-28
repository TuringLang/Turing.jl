---
title: Interface Developer Guide
toc: true
---

# The sampling interface

Turing implements a sampling interface that is intended to provide a common framework for Markov Chain Monte Carlo samplers. The interface presents several structures and functions that one needs to overload in order to implement an interface-compatible sampler. 

This guide will demonstrate how to implement the interface without Turing, and then demonstrate how to implement the same model with Turing.

## Interface overview

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

## Why do you have an interface?

The motivation for the interface is to allow Julia's fantastic probabilistic programming language community to have a set of standards and common implementations so we can all thrive together. Markov-Chain Monte Carlo methods tend to have a very similar framework to one another, and so a common interface should help more great inference methods built in single-purpose packages to experience more use among the community. 

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

## Metropolis-Hastings and Turing

If you are more interested in how you can build your sampler to use Turing's tools, this is the section for you. Turing provides an API for interacting with it's modelling language that is typically highly performant.

The first major difference between the vanilla interface implementation in the prior section and this section is the `VarInfo` type. `VarInfo` is a struct that contains all the information needed to manipulate a model -- it stores parameter values, distributions, log densities, and performs various record-keeping functions to make sure that the samplers can cooperate with one another. By default, all the parameters in `VarInfo` are initialized from the prior distributions. `VarInfo` is the workhorse of Turing and one of the major determinants of Turing's speed. The inclusion of this particularly struct changes the structure of the sampler slightly. Importantly, we will use a combination of immutable `Transition` structs and the mutable `VarInfo`.

Second, variables in `VarInfo` can be accessed using another type called a `VarName`, which refers to a specific variable in a `VarInfo`. As an example, if you have a `VarInfo` called `vi` and a `VarName` called `vn`, you can call `vi[vn]` to retrieve the value of the `VarName` that is currently store in `vi`. This can be set as well using `vi[vn] = foo`. We won't work too much with this in our sampler implementation, but this kind of thing is happening all the time in the background. 

Third, you no longer explicitly construct an `AbstractSampler` -- Turing has a default `Sampler` that stores an `InferenceAlgorithm`.

Fourth, instead of manually generating a proposal the way before, we will be using the `assume` interface. Turing uses an `observe`/`assume` style, where provided data is `observe`d and parameters are `assume`d. When you declare a model with `@model`, every line with a `~` is replaced with `observe` or `assume`. We'll get into that a little more in the following sections.

Fifth, you don't have to have a `Chains` function anymore. Turing has a good default one.

### Imports

As before, let's import the libraries and functions we'll need.

```julia
# Import the relevant libraries.
using Turing
using Turing.Interface
using Distributions

# Import specific functions and types to use or overload.
import Distributions: VariateForm, ValueSupport, variate_form, value_support, Sampleable, insupport
import MCMCChains: Chains
import Turing: Model, Sampler, Selector
import Turing.Interface: step!, AbstractTransition, transition_type
import Turing.Inference: InferenceAlgorithm, Transition, parameters!, 
                         getspace, assume, parameter
import Turing.RandomVariables: VarInfo, VarName, variables
```

The new functions we have imported from `Turing.Inference` include

- `InferenceAlgorithm`, which is a generic term for a struct that is stored inside a sampler. In our case, we'll be defining `MetropolisHastings <: InferenceAlgorithm`.
- `Transition` is Turing's default `AbstractTransition` struct, that contains a parameterization in the field `θ` and the log density of that parameterization in a field `lp`.
- `parameters!` is a function with the signature `parameters!(spl::Sampler, t::Transition)`, which accepts a `Transition` and updates the `VarInfo` to contain the parameters in the `Transition`. It returns a vector of the parameters if you would prefer to work with vectors instead of `NamedTuples`, which is how paramters are stored in a `Transition`.
- `getspace`, which defines the symbols that a sampler is allowed to operate on. This is required for all new `InferenceAlgorithm` types.
- `assume` is the function we have to overload to allow for custom `assume` statements.
- `parameter` accepts a `Transition` and a `VarName`, and returns the value of the variable stored in `Transition`.

We have also imported a couple of things from `Turing.RandomVariables`, which is the module containing all the tools for working with a `VarInfo` struct:

- `VarInfo` imports definitions for the `VarInfo` struct.
- `VarName` imports definitions for the `VarName` struct.
- `variables(spl::Sampler)` is a function that returns a vector containing the `VarNames` of a sampler, so you can retrieve or set information about them in a `VarInfo`.

### `InferenceAlgorithm`

```julia
# Define an InferenceAlgorithm type.
struct MetropolisHastings{space} <: InferenceAlgorithm 
    proposal :: Function
end

# Default constructors.
MetropolisHastings(space=Tuple{}()) = MetropolisHastings{space}(x -> Normal(x, 1))
MetropolisHastings(f::Function, space=Tuple{}()) = MetropolisHastings{space}(f)

# These functions are required for your sampler to function with Turing,
# and they return the variables that a sampler has ownership of.
getspace(::MetropolisHastings{space}) where {space} = space
getspace(::Type{<:MetropolisHastings{space}}) where {space} = space
```

We define a subtype of `InferenceAlgorithm` called `MetropolisHastings`. By default, Turing will bundle this up into a `Sampler` for us and give us a default `state` struct that carries a `VarInfo`.

`MetropolisHastings` accepts a custom proposal function, which returns a proposal distribution to draw from. By default, it will use a `Normal(x, 1)` proposal distribution. It also accepts a tuple called `space`, which is used if you want to only sample a subset of the variables in your model.

`getspace` returns the value of `space` -- if you build your own inference algorithm, you can just copy these two functions and replace `MetropolisHastings` with the name of your inference algorithm.

### `transition_type`

We need to tell the interface what type our `Transition` is going to be, and in this case we know exactly what the type is because of how handy the `VarInfo` is.

```julia
# Tell the interface what transition type we would like to use. We can use the default
# Turing.Inference.Transition struct, and the Transition(spl) functions it 
# provides.
function transition_type(model::Model, spl::Sampler{<:MetropolisHastings})
    return typeof(Transition(spl))
end
```

`Transition(spl)` simply takes the values in `spl.state.vi` and bundles them up into a `Transition` for later use.

### `proposal` and `assume`

Since our model definitions can be significantly more complex in Turing-land, we need to have a more robust `proposal` function.

```julia
# Define a function that makes a basic proposal. This function runs the model and
# bundles the results up. In this case, the actual proposal occurs during
# the assume function.
function proposal(spl::Sampler{<:MetropolisHastings}, model::Model, t::Transition)
    return Transition(model, spl, parameters!(spl, t))
end
```

This function takes the model, a sampler, and a vector of parameters (`parameters!` returns the vector of parameters that `t` contains). `Transition` will then run the model with the given parameters, and generate a new `Transition` containing the parameterization. The proposal is now generated in the `assume` statement:

```julia
function assume(
    spl::Sampler{<:MetropolisHastings},
    dist::Distribution,
    vn::VarName,
    vi::VarInfo
)
    # Retrieve the current parameter value.
    old_r = vi[vn]

    # Generate a proposal value.
    r = rand(spl.alg.proposal(vi[vn]))

    # Check if the proposal is in the distribution's support.
    if insupport(dist, r)
        # If the value is good, make sure to store it back in the VarInfo.
        vi[vn] = [r]
        return r, logpdf(dist, r)
    else
        # Otherwise return the previous value.
        return old_r, logpdf(dist, old_r)
    end
end
```

`assume` does the following for all parameters:

1. Get the value for each parameter.
2. Make a proposal distribution using the function stored in `spl`, and draw a random value from it.
3. Check if the new value is in the support of the parameter's distribution. This is necessary for constrained distributions like `Beta` or `InverseGamma` -- a more sophisticated `assume` statement would only draw from the bounds of the distribution, but for right now we'll keep it simple.
    a. If the new value is in the support, set the parameter in the `VarInfo` to be the proposal, and return a tuple with the draw and it's density.
    b. Otherwise, return the old value and it's density. This is an implicit sample rejection.

### `step!`

The step functions are almost entirely unchanged from the vanilla interface implementation:

```julia
# Define the first step! function, which is called the 
# beginning of sampling. Return the initial parameter used
# to define the sampler.
function step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:MetropolisHastings},
    N::Integer;
    kwargs...
)
    return Transition(spl)
end

# Define the other step functions. Returns a Transition containing
# either a new proposal (if accepted) or the previous proposal 
# (if not accepted).
function step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:MetropolisHastings},
    ::Integer,
    θ_prev::T;
    kwargs...
) where {
    T <: Transition
}
    # Generate a new proposal.
    θ = proposal(spl, model, θ_prev)
    
    # Calculate the log acceptance probability.
    α = θ.lp - θ_prev.lp + q(spl, θ_prev, θ)

    # Decide whether to return the previous θ or the new one.
    if log(rand()) < min(α, 0.0)
        return θ
    else
        return θ_prev
    end
end
```

The biggest different is the log acceptance probability calculation here:

```julia
# Calculate the log acceptance probability.
α = θ.lp - θ_prev.lp + q(spl, θ_prev, θ)
```

`q` now calculates whole proposal ratio instead of the numerator and denominator separately.

### `q`

The `q` function is now the last piece of the puzzle:

```julia
# Calculate the logpdf ratio of one proposal given another proposal.
function q(spl::Sampler{<:MetropolisHastings}, t1::Transition, t2::Transition)
    # Preallocate the ratio.
    ratio = 0.0

    # Iterate through each variable in the sampler.
    for vn in variables(spl)
        # Get the parameter from the Transition and the distribution 
        # associated with each variable.
        p1 = parameter(t1, vn)
        d1 = spl.alg.proposal(p1)

        p2 = parameter(t2, vn)
        d2 = spl.alg.proposal(p2)

        # Increment the log ratio.
        ratio += logpdf(d2, p1) - logpdf(d1, p2)
    end

    return ratio
end
```

This function accepts a sampler and two different transitions. It then iterates through the variables sotres in the sampler, retrieves the values stored in each `Transition`, calculates proposal distributions for each, and then aggregates their log densities.

### Testing our algorithm

Testing this algorithm is the same as using any other Turing model. Here, we'll use the `gdemo` model to determine the mean and standard deviation of a normal distribution.

```julia
# Model declaration.
@model gdemo(xs) = begin
    σ ~ InverseGamma(2,3)
    μ ~ Normal(0, sqrt(σ))
    for i in 1:length(xs)
        xs[i] ~ Normal(μ, σ)
    end
end

# Generate a set of data from the posterior we want to estimate.
data = rand(Normal(5,3), 50)

# Construct a DensityModel.
model = gdemo(data)

# Set up our sampler. Normal(x, 1.0) by default.
spl = MetropolisHastings()

# Sample from the posterior.
chain = sample(model, spl, 100000)
```

Once the sampling is complete, we end up with the following results:

```
Object of type Chains, with data of type 100000×3×1 Array{Union{Missing, Real},3}

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
│ 1   │ μ          │ 4.22902 │ 0.526566 │ 0.00166515 │ 0.00518738 │ 10453.9 │ 1.00019 │
│ 2   │ σ          │ 3.80514 │ 0.386095 │ 0.00122094 │ 0.00337472 │ 13253.7 │ 1.00011 │

Quantiles

│ Row │ parameters │ 2.5%    │ 25.0%   │ 50.0%   │ 75.0%   │ 97.5%   │
│     │ Symbol     │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │
├─────┼────────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ 1   │ μ          │ 3.1806  │ 3.89145 │ 4.23491 │ 4.57381 │ 5.24438 │
│ 2   │ σ          │ 3.14229 │ 3.5346  │ 3.76932 │ 4.04328 │ 4.66065 │
```

Looks good to me! 

## Conclusion

We've seen how to implement the sampling interface in general terms (for those who don't want Turing's great tools) and in Turing (for those who love great tools). The developer interface is ever-improving, so please open an [issue on GitHub](https://github.com/TuringLang/Turing.jl/issues) with any bugs or feature requests.