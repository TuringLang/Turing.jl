# The Sampling Interface

Turing's sampling interface presents several structures and functions that one needs to overload in order to implement an interface-compatible sampler.

1. A subtype of `AbstractSampler`, defined as a mutable struct containing state information.
2. A subtype of `AbstractTransition`, which represents a single draw from the sampler.
3. A function `transition_type` which returns the `AbstractTransition` type used by an implementation of an `AbstractSampler`, or a function `transition_init` with returns a `Vector{AbstractTransition}` of length `N`.
4. A function `sample_init!` which performs any necessary set up. 
5. A function `step!` which returns an `AbstractTransition`.
6. A function `sample_end!` which handles any sampler wrap-up.
7. A function `MCMCChains.Chains` which accepts an `Vector{<:AbstractTransition}` and returns an `MCMCChains` object.

The interface methods with exclamation points are those that are intended to allow for some state mutation. Any mutating function is meant to allow mutation where needed -- you might use 

- `sample_init!` to run some kind of sampler preparation, before sampling begins. This could mutate a sampler's state.
- `step!` might mutate a sampler flag after each sample. MH does this for example by using a `violating_support` flag.
- `sample_end!` contains any wrap-up you might need to do. If you were sampling in a transformed space, this might be where you convert everything back to a constrained space.