"""
    InferenceAlgorithm

Abstract type representing an inference algorithm in Turing. Note that this is
not the same as an `AbstractSampler`: the latter is what defines the necessary
methods for actually sampling.

To create an `AbstractSampler`, the `InferenceAlgorithm` needs to be wrapped in
`DynamicPPL.Sampler`. If `sample()` is called with an `InferenceAlgorithm`,
this wrapping occurs automatically.
"""
abstract type InferenceAlgorithm end

DynamicPPL.default_chain_type(::Sampler{<:InferenceAlgorithm}) = MCMCChains.Chains

# Overloaded by adaptive Hamiltonians
# The second argument is the number of samples requested
default_num_warmup(::InferenceAlgorithm, ::Int) = 0
