
<a id='The-Sampling-Interface-1'></a>

# The Sampling Interface


Turing's sampling interface presents several structures and functions that one needs to overload in order to implement an interface-compatible sampler. Currently it is non-operational, but the intent is to have all of Turing's samplers using the API below.


1. A subtype of `AbstractSampler`, defined as a mutable struct containing state information
2. A subtype of `AbstractTransition`, which represents a single draw from the sampler
3. A function `transitions_init` which returns a preallocated vector of the `AbstractTransition` type used by an implementation of an `AbstractSampler`
4. A function `sample_init!` which performs any necessary set up
5. A function `step!` which returns an `AbstractTransition`
6. A function `sample_end!` which handles any sampler wrap-up
7. A function `MCMCChains.Chains` which accepts the post-sampling vector created by `transitions_init` returns an `MCMCChains` object

