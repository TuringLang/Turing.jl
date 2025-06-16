# TODO(penelopeysm): remove
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

# TODO(penelopeysm): remove
DynamicPPL.default_chain_type(sampler::Sampler{<:InferenceAlgorithm}) = MCMCChains.Chains

"""
    update_sample_kwargs(spl::AbstractSampler, N::Integer, kwargs)

Some samplers carry additional information about the keyword arguments that
should be passed to `AbstractMCMC.sample`. This function provides a hook for
them to update the default keyword arguments. The default implementation is for
no changes to be made to `kwargs`.
"""
update_sample_kwargs(::AbstractSampler, N::Integer, kwargs) = kwargs

"""
    get_adtype(spl::AbstractSampler)

Return the automatic differentiation (AD) backend to use for the sampler. This
is needed for constructing a LogDensityFunction. By default, returns nothing,
i.e. the LogDensityFunction that is constructed will not know how to calculate
its gradients. If the sampler requires gradient information, then this function
must return an `ADTypes.AbstractADType`.
"""
get_adtype(::AbstractSampler) = nothing

"""
    requires_unconstrained_space(sampler::AbstractSampler)

Return `true` if the sampler / algorithm requires unconstrained space, and
`false` otherwise. This is used to determine whether the initial VarInfo
should be linked. Defaults to true.
"""
requires_unconstrained_space(::AbstractSampler) = true
