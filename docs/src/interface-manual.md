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

Let's begin our sampler definition by defining a sampler called `MetropolisHastings` which is a subtype of `AbstractSampler`. Correct typing is very important for proper interface implementation -- if you are missing a subtype, your inference method will likely not work when you call `sample`.

```julia
# Define a sampler type.
struct MetropolisHastings{T} <: AbstractSampler 
    init_θ :: T
end
```

Above, we have defined a sampler that only stores the initial parameterization of the prior. You can have a struct that has no fields, and simply use it for dispatching onto the relevant functions, or you can store a large amount of state information in your sampler. This implementation of Metropolis-Hastings is a pure immutable one, and no state information need be saved. 

The general intuition for what to store in your sampler struct is that anything you may need to perform inference between samples but you don't want to store in an `AbstractTransition` should go into the sampler struct. It's the only way you can carry non-sample related state information between `step!` calls. 

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

The next step is to define some subtype of `AbstractTransition` which we will return from each `step!` call. We'll keep it simple by just defining a wrapper struct that holds only the parameter draws:

```julia
# Create a very basic Transition type, only stores the parameter draws.
struct Transition{T} <: AbstractTransition
    θ :: T
end
```

`Transition` can now store any type, whether it's a vector of draws or a single univariate draw. Again, this could be improved by adding type constraints to `T`, but for this simple sampler we are largely unconcerned with 