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

function DynamicPPL.init_strategy(sampler::Sampler{<:InferenceAlgorithm})
    return DynamicPPL.InitFromPrior()
end
