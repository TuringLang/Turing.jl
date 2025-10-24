"""
    ExternalSampler{Unconstrained,S<:AbstractSampler,AD<:ADTypes.AbstractADType}

Represents a sampler that does not have a custom implementation of `AbstractMCMC.step(rng,
::DynamicPPL.Model, spl)`.

The `Unconstrained` type-parameter is to indicate whether the sampler requires unconstrained space.

# Fields
$(TYPEDFIELDS)

# Turing.jl's interface for external samplers

If you implement a new `MySampler <: AbstractSampler` and want it to work with Turing.jl
models, there are two options:

1. Directly implement the `AbstractMCMC.step` methods for `DynamicPPL.Model`. That is to
   say, implement `AbstractMCMC.step(rng::Random.AbstractRNG, model::DynamicPPL.Model,
   sampler::MySampler; kwargs...)` and related methods. This is the most powerful option and
   is what Turing.jl's in-house samplers do. Implementing this means that you can directly
   call `sample(model, MySampler(), N)`.

2. Implement a generic `AbstractMCMC.step` method for `AbstractMCMC.LogDensityModel` (the
   same signature as above except that `model::AbstractMCMC.LogDensityModel`). This struct
   wraps an object that obeys the LogDensityProblems.jl interface, so your `step`
   implementation does not need to know anything about Turing.jl or DynamicPPL.jl. To use
   this with Turing.jl, you will need to wrap your sampler: `sample(model,
   externalsampler(MySampler()), N)`.

This section describes the latter.

`MySampler` **must** implement the following methods:

- `AbstractMCMC.step` (the main function for taking a step in MCMC sampling; this is
  documented in AbstractMCMC.jl). This function must return a tuple of two elements, a
  'transition' and a 'state'.

- `AbstractMCMC.getparams(external_state)`: How to extract the parameters from the **state**
  returned by your sampler (i.e., the **second** return value of `step`). For your sampler
  to work with Turing.jl, this function should return a Vector of parameter values. Note that
  this function does not need to perform any linking or unlinking; Turing.jl will take care of
  this for you. You should return the parameters *exactly* as your sampler sees them.

- `AbstractMCMC.getstats(external_state)`: Extract sampler statistics corresponding to this
  iteration from the **state** returned by your sampler (i.e., the **second** return value
  of `step`). For your sampler to work with Turing.jl, this function should return a
  `NamedTuple`. If there are no statistics to return, return `NamedTuple()`.

  Note that `getstats` should not include log-probabilities as these will be recalculated by
  Turing automatically for you.

Notice that both of these functions take the **state** as input, not the **transition**. In
other words, the transition is completely useless for the external sampler interface. This is
in line with long-term plans for removing transitions from AbstractMCMC.jl and only using
states.

There are a few more optional functions which you can implement to improve the integration
with Turing.jl:

- `AbstractMCMC.requires_unconstrained_space(::MySampler)`: If your sampler requires
  unconstrained space, you should return `true`. This tells Turing to perform linking on the
  VarInfo before evaluation, and ensures that the parameter values passed to your sampler
  will always be in unconstrained (Euclidean) space.

- `Turing.Inference.isgibbscomponent(::MySampler)`: If you want to disallow your sampler
  from a component in Turing's Gibbs sampler, you should make this evaluate to `false`. Note
  that the default is `true`, so you should only need to implement this in special cases.
"""
struct ExternalSampler{Unconstrained,S<:AbstractSampler,AD<:ADTypes.AbstractADType} <:
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
    - `unconstrained::Val`: Value type containing a boolean indicating whether the sampler requires unconstrained space.
    """
    function ExternalSampler(
        sampler::AbstractSampler, adtype::ADTypes.AbstractADType, ::Val{unconstrained}
    ) where {unconstrained}
        if !(unconstrained isa Bool)
            throw(
                ArgumentError("Expected Val{true} or Val{false}, got Val{$unconstrained}")
            )
        end
        return new{unconstrained,typeof(sampler),typeof(adtype)}(sampler, adtype)
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
function externalsampler(sampler::AbstractSampler; adtype=Turing.DEFAULT_ADTYPE)
    return ExternalSampler(
        sampler, adtype, Val(AbstractMCMC.requires_unconstrained_space(sampler))
    )
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

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler_wrapper::ExternalSampler{unconstrained};
    initial_state=nothing,
    initial_params, # passed through from sample
    kwargs...,
) where {unconstrained}
    sampler = sampler_wrapper.sampler

    # Initialise varinfo with initial params and link the varinfo if needed.
    varinfo = DynamicPPL.VarInfo(model)
    _, varinfo = DynamicPPL.init!!(rng, model, varinfo, initial_params)

    if unconstrained
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
    _, state_inner = if initial_state === nothing
        AbstractMCMC.step(
            rng,
            AbstractMCMC.LogDensityModel(f),
            sampler;
            initial_params=initial_params_vector,
            kwargs...,
        )

    else
        AbstractMCMC.step(
            rng,
            AbstractMCMC.LogDensityModel(f),
            sampler,
            initial_state;
            initial_params=initial_params_vector,
            kwargs...,
        )
    end

    new_parameters = AbstractMCMC.getparams(f.model, state_inner)
    new_vi = DynamicPPL.unflatten(f.varinfo, new_parameters)
    new_stats = AbstractMCMC.getstats(state_inner)
    return (
        Turing.Inference.Transition(f.model, new_vi, new_stats),
        TuringState(state_inner, new_vi, f),
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
    _, state_inner = AbstractMCMC.step(
        rng, AbstractMCMC.LogDensityModel(f), sampler, state.state; kwargs...
    )

    new_parameters = AbstractMCMC.getparams(f.model, state_inner)
    new_vi = DynamicPPL.unflatten(f.varinfo, new_parameters)
    new_stats = AbstractMCMC.getstats(state_inner)
    return (
        Turing.Inference.Transition(f.model, new_vi, new_stats),
        TuringState(state_inner, new_vi, f),
    )
end

# Implementation of interface for AdvancedMH and AdvancedHMC. TODO: These should be
# upstreamed to the respective packages, I'm just not doing it here to avoid having to run
# CI against three separate PR branches.
AbstractMCMC.getstats(state::AdvancedHMC.HMCState) = state.transition.stat
# Note that for AdvancedMH, transition and state are equivalent (and both named Transition)
AbstractMCMC.getstats(state::AdvancedMH.Transition) = (accepted=state.accepted,)
