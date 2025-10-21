"""
    ExternalSampler{S<:AbstractSampler,AD<:ADTypes.AbstractADType,Unconstrained}

Represents a sampler that does not have a custom implementation of `AbstractMCMC.step(rng,
::DynamicPPL.Model, spl)`.

The `Unconstrained` type-parameter is to indicate whether the sampler requires unconstrained space.

# Fields
$(TYPEDFIELDS)

# Turing.jl's interface for external samplers

If you implement a new `MySampler <: AbstractSampler` and want it to work with Turing.jl
models, there are two options:

1. Directly implement the `AbstractMCMC.step` methods for `DynamicPPL.Model`. This is the
   most powerful option and is what Turing.jl's in-house samplers do. Implementing this
   means that you can directly call `sample(model, MySampler(), N)`.

2. Implement a generic `AbstractMCMC.step` method for `AbstractMCMC.LogDensityModel`. This
   struct wraps an object that obeys the LogDensityProblems.jl interface, so your `step`
   implementation does not need to know anything about Turing.jl or DynamicPPL.jl. To use
   this with Turing.jl, you will need to wrap your sampler: `sample(model,
   externalsampler(MySampler()), N)`.

This section describes the latter.

`MySampler` must implement the following methods:

- `AbstractMCMC.step` (the main function for taking a step in MCMC sampling; this is
  documented in AbstractMCMC.jl)
- `Turing.Inference.getparams(::DynamicPPL.Model, external_transition)`: How to extract the
  parameters from the transition returned by your sampler (i.e., the first return value of
  `step`). There is a default implementation for this method, which is to return
  `external_transition.θ`.

!!! note
    In a future breaking release of Turing, this is likely to change to
    `AbstractMCMC.getparams(::DynamicPPL.Model, external_state)`, with no default method.
    `Turing.Inference.getparams` is technically an internal method, so the aim here is to
    unify the interface for samplers at a higher level.

There are a few more optional functions which you can implement to improve the integration
with Turing.jl:

- `Turing.Inference.isgibbscomponent(::MySampler)`: If you want your sampler to function as
  a component in Turing's Gibbs sampler, you should make this evaluate to `true`.

- `Turing.Inference.requires_unconstrained_space(::MySampler)`: If your sampler requires
  unconstrained space, you should return `true`. This tells Turing to perform linking on the
  VarInfo before evaluation, and ensures that the parameter values passed to your sampler
  will always be in unconstrained (Euclidean) space.
"""
struct ExternalSampler{S<:AbstractSampler,AD<:ADTypes.AbstractADType,Unconstrained} <:
       AbstractSampler
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
    sampler_wrapper::ExternalSampler;
    initial_state=nothing,
    initial_params, # passed through from sample
    kwargs...,
)
    sampler = sampler_wrapper.sampler

    # Initialise varinfo with initial params and link the varinfo if needed.
    varinfo = DynamicPPL.VarInfo(model)
    _, varinfo = DynamicPPL.init!!(rng, model, varinfo, initial_params)

    if requires_unconstrained_space(sampler_wrapper)
        varinfo = DynamicPPL.link(varinfo, model)
    end

    # We need to extract the vectorised initial_params, because the later call to
    # AbstractMCMC.step only sees a `LogDensityModel` which expects `initial_params`
    # to be a vector.
    initial_params_vector = varinfo[:]

    # Construct LogDensityFunction
    f = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getlogjoint_internal, varinfo; adtype=sampler_wrapper.adtype
    )

    # Then just call `AbstractMCMC.step` with the right arguments.
    if initial_state === nothing
        transition_inner, state_inner = AbstractMCMC.step(
            rng,
            AbstractMCMC.LogDensityModel(f),
            sampler;
            initial_params=initial_params_vector,
            kwargs...,
        )
    else
        transition_inner, state_inner = AbstractMCMC.step(
            rng,
            AbstractMCMC.LogDensityModel(f),
            sampler,
            initial_state;
            initial_params=initial_params_vector,
            kwargs...,
        )
    end

    # NOTE: This is Turing.Inference.getparams, not AbstractMCMC.getparams (!!!!!)
    # The latter uses the state rather than the transition.
    # TODO(penelopeysm): Make this use AbstractMCMC.getparams instead
    new_parameters = Turing.Inference.getparams(f.model, transition_inner)
    new_vi = DynamicPPL.unflatten(f.varinfo, new_parameters)
    return (
        Transition(f.model, new_vi, transition_inner), TuringState(state_inner, new_vi, f)
    )
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler_wrapper::ExternalSampler,
    state::TuringState;
    kwargs...,
)
    sampler = sampler_wrapper.sampler
    f = state.ldf

    # Then just call `AdvancedMCMC.step` with the right arguments.
    transition_inner, state_inner = AbstractMCMC.step(
        rng, AbstractMCMC.LogDensityModel(f), sampler, state.state; kwargs...
    )

    # NOTE: This is Turing.Inference.getparams, not AbstractMCMC.getparams (!!!!!)
    # The latter uses the state rather than the transition.
    # TODO(penelopeysm): Make this use AbstractMCMC.getparams instead
    new_parameters = Turing.Inference.getparams(f.model, transition_inner)
    new_vi = DynamicPPL.unflatten(f.varinfo, new_parameters)
    return (
        Transition(f.model, new_vi, transition_inner), TuringState(state_inner, new_vi, f)
    )
end
