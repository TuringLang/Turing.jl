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
- `Turing.Inference.getparams(::DynamicPPL.Model, external_transition)`: How to extract the parameters from the transition returned by your sampler (i.e., the first return value of `step`).
  There is a default implementation for this method, which is to return `external_transition.θ`.

!!! note
    In a future breaking release of Turing, this is likely to change to `AbstractMCMC.getparams(::DynamicPPL.Model, external_state)`, with no default method. `Turing.Inference.getparams` is technically an internal method, so the aim here is to unify the interface for samplers at a higher level.

There are a few more optional functions which you can implement to improve the integration with Turing.jl:

- `Turing.Inference.isgibbscomponent(::MySampler)`: If you want your sampler to function as a component in Turing's Gibbs sampler, you should make this evaluate to `true`.

- `Turing.Inference.requires_unconstrained_space(::MySampler)`: If your sampler requires unconstrained space, you should return `true`. This tells Turing to perform linking on the VarInfo before evaluation, and ensures that the parameter values passed to your sampler will always be in unconstrained (Euclidean) space.

- `Turing.Inference.getlogp_external(external_transition, external_state)`: Tell Turing how to extract the log probability density associated with this transition (and state). If you do not specify these, Turing will simply re-evaluate the model with the parameters obtained from `getparams`, which can be inefficient. It is therefore recommended to store the log probability density in either the transition or the state (or both) and override this method.
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

"""
    getlogp_external(external_transition, external_state)

Get the log probability density associated with the external sampler's
transition and state. Returns `missing` by default; in this case, an extra
model evaluation will be needed to calculate the correct log density.
"""
getlogp_external(::Any, ::Any) = missing
getlogp_external(mh::AdvancedMH.Transition, ::AdvancedMH.Transition) = mh.lp
getlogp_external(hmc::AdvancedHMC.Transition, ::AdvancedHMC.HMCState) = hmc.stat.log_density

struct TuringState{S,V1<:AbstractVarInfo,M,V,C}
    state::S
    # Note that this varinfo has the correct parameters and logp obtained from
    # the state, whereas `ldf.varinfo` will in general have junk inside it.
    varinfo::V1
    ldf::DynamicPPL.LogDensityFunction{M,V,C}
end

varinfo(state::TuringState) = state.varinfo
varinfo(state::AbstractVarInfo) = state

getparams(::DynamicPPL.Model, transition::AdvancedHMC.Transition) = transition.z.θ
function getparams(model::DynamicPPL.Model, state::AdvancedHMC.HMCState)
    return getparams(model, state.transition)
end
getstats(transition::AdvancedHMC.Transition) = transition.stat

getparams(::DynamicPPL.Model, transition::AdvancedMH.Transition) = transition.params

function make_updated_varinfo(
    f::DynamicPPL.LogDensityFunction, external_transition, external_state
)
    # Set the parameters.
    # NOTE: This is Turing.Inference.getparams, not AbstractMCMC.getparams (!!!!!)
    # The latter uses the state rather than the transition.
    # TODO(penelopeysm): Make this use AbstractMCMC.getparams instead
    new_parameters = getparams(f.model, external_transition)
    new_varinfo = DynamicPPL.unflatten(f.varinfo, new_parameters)
    # Set (or recalculate, if needed) the log density.
    new_logp = getlogp_external(external_transition, external_state)
    return if ismissing(new_logp)
        last(DynamicPPL.evaluate!!(f.model, new_varinfo, f.context))
    else
        DynamicPPL.setlogp!!(new_varinfo, new_logp)
    end
end

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

    # Get the parameters and log density, and set them in the varinfo.
    new_varinfo = make_updated_varinfo(f, transition_inner, state_inner)

    # Update the `state`
    return (
        Transition(f.model, new_varinfo, transition_inner),
        TuringState(state_inner, new_varinfo, f),
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

    # Get the parameters and log density, and set them in the varinfo.
    new_varinfo = make_updated_varinfo(f, transition_inner, state_inner)

    # Update the `state`
    return (
        Transition(f.model, new_varinfo, transition_inner),
        TuringState(state_inner, new_varinfo, f),
    )
end
