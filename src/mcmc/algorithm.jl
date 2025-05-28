"""
    InferenceAlgorithm

Abstract type representing an inference algorithm in Turing. Note that this is
not the same as an `AbstractSampler`: the latter is what defines the necessary
methods for actually sampling.

To create an `AbstractSampler`, the `InferenceAlgorithm` needs to be wrapped
in `DynamicPPL.Sampler`. If `sample()` is called with an `InferenceAlgorithm`,
this wrapping occurs automatically.
"""
abstract type InferenceAlgorithm end

"""
    update_sample_kwargs(spl::AbstractSampler, N::Integer, kwargs)
    update_sample_kwargs(spl::InferenceAlgorithm, N::Integer, kwargs)

Some InferenceAlgorithm implementations carry additional information about
the keyword arguments that should be passed to `AbstractMCMC.sample`. This
function provides a hook for them to update the default keyword arguments.

The default implementation is for no changes to be made to `kwargs`.
"""
function update_sample_kwargs(spl::Sampler{<:InferenceAlgorithm}, N::Integer, kwargs)
    return update_sample_kwargs(spl.alg, N, kwargs)
end
update_sample_kwargs(::AbstractSampler, N::Integer, kwargs) = kwargs
update_sample_kwargs(::InferenceAlgorithm, N::Integer, kwargs) = kwargs

"""
    get_adtype(spl::AbstractSampler)
    get_adtype(spl::InferenceAlgorithm)

Return the automatic differentiation (AD) backend to use for the sampler.
This is needed for constructing a LogDensityFunction.

By default, returns nothing, i.e. the LogDensityFunction that is constructed
will not know how to calculate its gradients.

If the sampler or algorithm requires gradient information, then this function
must return an `ADTypes.AbstractADType`.
"""
get_adtype(::AbstractSampler) = nothing
get_adtype(::InferenceAlgorithm) = nothing
get_adtype(spl::Sampler{<:InferenceAlgorithm}) = get_adtype(spl.alg)

"""
    requires_unconstrained_space(sampler::AbstractSampler)
    requires_unconstrained_space(sampler::InferenceAlgorithm)

Return `true` if the sampler / algorithm requires unconstrained space, and
`false` otherwise. This is used to determine whether the initial VarInfo
should be linked. Defaults to true.
"""
requires_unconstrained_space(::AbstractSampler) = true
requires_unconstrained_space(::InferenceAlgorithm) = true
function requires_unconstrained_space(spl::Sampler{<:InferenceAlgorithm})
    return requires_unconstrained_space(spl.alg)
end
