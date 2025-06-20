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
        (::Val{unconstrained})=Val(true),
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
    requires_unconstrained_space(sampler::ExternalSampler)

Return `true` if the sampler requires unconstrained space, and `false` otherwise.
"""
function requires_unconstrained_space(
    ::ExternalSampler{<:Any,<:Any,Unconstrained}
) where {Unconstrained}
    Unconstrained
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

struct TuringState{S,M,V,C}
    state::S
    ldf::DynamicPPL.LogDensityFunction{M,V,C}
end

state_to_turing(f::DynamicPPL.LogDensityFunction, state) = TuringState(state, f)
function transition_to_turing(f::DynamicPPL.LogDensityFunction, transition)
    # TODO: We should probably rename this `getparams` since it returns something
    # very different from `Turing.Inference.getparams`.
    θ = getparams(f.model, transition)
    varinfo = DynamicPPL.unflatten(f.varinfo, θ)
    return Transition(f.model, varinfo, transition)
end

function varinfo(state::TuringState)
    θ = getparams(state.ldf.model, state.state)
    # TODO: Do we need to link here first?
    return DynamicPPL.unflatten(state.ldf.varinfo, θ)
end
varinfo(state::AbstractVarInfo) = state

# NOTE: Only thing that depends on the underlying sampler.
# Something similar should be part of AbstractMCMC at some point:
# https://github.com/TuringLang/AbstractMCMC.jl/pull/86
getparams(::DynamicPPL.Model, transition::AdvancedHMC.Transition) = transition.z.θ
function getparams(model::DynamicPPL.Model, state::AdvancedHMC.HMCState)
    return getparams(model, state.transition)
end
getstats(transition::AdvancedHMC.Transition) = transition.stat

getparams(::DynamicPPL.Model, transition::AdvancedMH.Transition) = transition.params

# TODO: Do we also support `resume`, etc?
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler_wrapper::Sampler{<:ExternalSampler};
    initial_state=nothing,
    initial_params=nothing,
    kwargs...,
)
    alg = sampler_wrapper.alg
    sampler = alg.sampler

    # Initialise varinfo with initial params and link the varinfo if needed.
    varinfo = DynamicPPL.VarInfo(model)
    if requires_unconstrained_space(alg)
        if initial_params !== nothing
            # If we have initial parameters, we need to set the varinfo before linking.
            varinfo = DynamicPPL.link(DynamicPPL.unflatten(varinfo, initial_params), model)
            # Extract initial parameters in unconstrained space.
            initial_params = varinfo[:]
        else
            varinfo = DynamicPPL.link(varinfo, model)
        end
    end

    # Construct LogDensityFunction
    f = DynamicPPL.LogDensityFunction(model, varinfo; adtype=alg.adtype)

    # Then just call `AbstractMCMC.step` with the right arguments.
    if initial_state === nothing
        transition_inner, state_inner = AbstractMCMC.step(
            rng, AbstractMCMC.LogDensityModel(f), sampler; initial_params, kwargs...
        )
    else
        transition_inner, state_inner = AbstractMCMC.step(
            rng,
            AbstractMCMC.LogDensityModel(f),
            sampler,
            initial_state;
            initial_params,
            kwargs...,
        )
    end
    # Update the `state`
    return transition_to_turing(f, transition_inner), state_to_turing(f, state_inner)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler_wrapper::Sampler{<:ExternalSampler},
    state::TuringState;
    kwargs...,
)
    sampler = sampler_wrapper.alg.sampler
    f = state.ldf

    # Then just call `AdvancedHMC.step` with the right arguments.
    transition_inner, state_inner = AbstractMCMC.step(
        rng, AbstractMCMC.LogDensityModel(f), sampler, state.state; kwargs...
    )

    # Update the `state`
    return transition_to_turing(f, transition_inner), state_to_turing(f, state_inner)
end
