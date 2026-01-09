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
    externalsampler(
        sampler::AbstractSampler;
        adtype=AutoForwardDiff(),
        unconstrained=AbstractMCMC.requires_unconstrained_space(sampler),
    )

Wrap a sampler so it can be used as an inference algorithm.

# Arguments
- `sampler::AbstractSampler`: The sampler to wrap.

# Keyword Arguments
- `adtype::ADTypes.AbstractADType=ADTypes.AutoForwardDiff()`: The automatic differentiation
  (AD) backend to use.
- `unconstrained::Bool=AbstractMCMC.requires_unconstrained_space(sampler)`: Whether the
  sampler requires unconstrained space.
"""
function externalsampler(
    sampler::AbstractSampler;
    adtype=Turing.DEFAULT_ADTYPE,
    unconstrained::Bool=AbstractMCMC.requires_unconstrained_space(sampler),
)
    return ExternalSampler(sampler, adtype, Val(unconstrained))
end

# TODO(penelopeysm): Can't we clean this up somehow?
struct TuringState{S,V,P<:AbstractVector,L<:DynamicPPL.LogDensityFunction}
    state::S
    # Note that this varinfo is used only for structure. Its parameters and other info do
    # not need to be accurate
    varinfo::V
    # These are the actual parameters that this state is at
    params::P
    ldf::L
end

# get_varinfo must return something from which the correct parameters can be obtained
function get_varinfo(state::TuringState)
    return DynamicPPL.unflatten(state.varinfo, state.params)
end
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


    # Construct LogDensityFunction FIRST (we need this for validation)
    f = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getlogjoint_internal, varinfo; adtype=sampler_wrapper.adtype
    )

    # Use shared function to find valid initial parameters with gradient checking
    validator = vi -> begin
        θ = vi[:]
        logp, grad = LogDensityProblems.logdensity_and_gradient(f, θ)
        return isfinite(logp) && all(isfinite, grad)

    end
    
    varinfo = find_initial_params(
        rng, model, varinfo, initial_params, validator; max_attempts=10
    )
    
    initial_params_vector = varinfo[:]


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
    new_stats = AbstractMCMC.getstats(state_inner)
    return (
        DynamicPPL.ParamsWithStats(new_parameters, f, new_stats),
        TuringState(state_inner, varinfo, new_parameters, f),
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
    new_stats = AbstractMCMC.getstats(state_inner)
    return (
        DynamicPPL.ParamsWithStats(new_parameters, f, new_stats),
        TuringState(state_inner, state.varinfo, new_parameters, f),
    )
end
