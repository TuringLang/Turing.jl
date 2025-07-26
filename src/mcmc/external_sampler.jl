"""
    ExternalSampler{S<:AbstractSampler,AD<:ADTypes.AbstractADType,Unconstrained}

Represents a sampler that is not an implementation of `InferenceAlgorithm`.

The `Unconstrained` type-parameter is to indicate whether the sampler requires unconstrained space.

# Fields
$(TYPEDFIELDS)

# Turing.jl's interface for external samplers

When implementing a new `MySampler <: AbstractSampler`,
`MySampler` must first and foremost conform to the `AbstractMCMC` interface to work with Turing.jl's `externalsampler` function.
In particular, it must implement:

- `AbstractMCMC.step` (the main function for taking a step in MCMC sampling; this is documented in AbstractMCMC.jl)
- `AbstractMCMC.getparams(::DynamicPPL.Model, external_state)`: How to extract the parameters from the state returned by your sampler (i.e., the second return value of `step`).

There are a few more optional functions which you can implement to improve the integration with Turing.jl:

- `Turing.Inference.isgibbscomponent(::MySampler)`: If you want your sampler to function as a component in Turing's Gibbs sampler, you should make this evaluate to `true`.

- `Turing.Inference.requires_unconstrained_space(::MySampler)`: If your sampler requires unconstrained space, you should return `true`. This tells Turing to perform linking on the VarInfo before evaluation, and ensures that the parameter values passed to your sampler will always be in unconstrained (Euclidean) space.
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
    return Unconstrained
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

# TODO(penelopeysm): Can't we clean this up somehow?
struct TuringState{S,V1,M,V}
    state::S
    # Note that this varinfo must have the correct parameters set; but logp
    # does not matter as it will be re-evaluated
    varinfo::V1
    # Note that in general the VarInfo inside this LogDensityFunction will have
    # junk parameters and logp. It only exists to provide structure
    ldf::DynamicPPL.LogDensityFunction{M,V}
end

# get_varinfo should return something from which the correct parameters can be
# obtained, hence we use state.varinfo rather than state.ldf.varinfo
get_varinfo(state::TuringState) = state.varinfo
get_varinfo(state::AbstractVarInfo) = state

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
    f = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getlogjoint_internal, varinfo; adtype=alg.adtype
    )

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

    new_parameters = getparams(f.model, state_inner)
    new_vi = DynamicPPL.unflatten(f.varinfo, new_parameters)
    return (
        Transition(f.model, new_vi, transition_inner), TuringState(state_inner, new_vi, f)
    )
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

    # Then just call `AdvancedMCMC.step` with the right arguments.
    transition_inner, state_inner = AbstractMCMC.step(
        rng, AbstractMCMC.LogDensityModel(f), sampler, state.state; kwargs...
    )

    new_parameters = getparams(f.model, state_inner)
    new_vi = DynamicPPL.unflatten(f.varinfo, new_parameters)
    return (
        Transition(f.model, new_vi, transition_inner), TuringState(state_inner, new_vi, f)
    )
end
