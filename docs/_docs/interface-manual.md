# The Sampling Interface

Turing's sampling interface presents several structures and functions that one needs to overload in order to implement an interface-compatible sampler. Currently it is non-operational, but the intent is to have all of Turing's samplers using the API below.

1. A subtype of `AbstractSampler`, defined as a mutable struct containing state information
2. A subtype of `AbstractTransition`, which represents a single draw from the sampler
3. A function `transition_type` which returns the `AbstractTransition` type used by an implementation of an `AbstractSampler`
4. A function `sample_init!` which performs any necessary set up
5. A function `step!` which returns an `AbstractTransition`
6. A function `sample_end!` which handles any sampler wrap-up
7. A function `MCMCChains.Chains` which accepts an `Array{AbstractTransition, 1}` and returns an `MCMCChains` object
