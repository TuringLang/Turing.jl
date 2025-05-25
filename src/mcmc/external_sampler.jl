"""
    ExternalSampler{S<:AbstractSampler,AD<:ADTypes.AbstractADType,Unconstrained}

Represents a sampler that is not an implementation of `InferenceAlgorithm`.

The `Unconstrained` type-parameter is to indicate whether the sampler requires unconstrained space.

# Fields
$(TYPEDFIELDS)
"""
struct ExternalSampler{S<:AbstractSampler,AD<:ADTypes.AbstractADType,Unconstrained} <:
       InferenceAlgorithm
    "the sampler to wrap"
    sampler::S
    "the automatic differentiation (AD) backend to use"
    adtype::AD

    """
        ExternalSampler(sampler::AbstractSampler, adtype::ADTypes.AbstractADType, ::Val{unconstrained})

    Wrap a sampler so it can be used as an inference algorithm.

    # Arguments
    - `sampler::AbstractSampler`: The sampler to wrap.
    - `adtype::ADTypes.AbstractADType`: The automatic differentiation (AD) backend to use.
    - `unconstrained::Val=Val{true}()`: Value type containing a boolean indicating whether the sampler requires unconstrained space.
    """
    function ExternalSampler(
        sampler::AbstractSampler,
        adtype::ADTypes.AbstractADType,
        ::Val{unconstrained}=Val(true),
    ) where {unconstrained}
        if !(unconstrained isa Bool)
            throw(
                ArgumentError("Expected Val{true} or Val{false}, got Val{$unconstrained}")
            )
        end
        return new{typeof(sampler),typeof(adtype),unconstrained}(sampler, adtype)
    end
end

"""
    externalsampler(sampler::AbstractSampler; adtype=AutoForwardDiff(), unconstrained=true)

Wrap a sampler so it can be used as an inference algorithm.

# Arguments
- `sampler::AbstractSampler`: The sampler to wrap.

# Keyword Arguments
- `adtype::ADTypes.AbstractADType=ADTypes.AutoForwardDiff()`: The automatic differentiation (AD) backend to use.
- `unconstrained::Bool=true`: Whether the sampler requires unconstrained space.
"""
function externalsampler(
    sampler::AbstractSampler; adtype=Turing.DEFAULT_ADTYPE, unconstrained::Bool=true
)
    return ExternalSampler(sampler, adtype, Val(unconstrained))
end

function requires_unconstrained_space(
    ::ExternalSampler{<:Any,<:Any,Unconstrained}
) where {Unconstrained}
    return Unconstrained
end

get_adtype(spl::ExternalSampler) = spl.adtype
