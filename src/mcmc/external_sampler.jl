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

- `AbstractMCMC.step_warmup` (optional; documented in AbstractMCMC.jl). If your sampler
  adapts, implement this and Turing.jl will route the `num_warmup` iterations through it.
  Samplers that do not implement it fall back to `AbstractMCMC.step`, as before.

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

- `Turing.Inference.supports_gibbs(::MySampler)`: If you want to disallow your sampler
  from a component in Turing's Gibbs sampler, you should make this evaluate to `false`. Note
  that the default is `true`, so you should only need to implement this in special cases.

- `Turing.Inference.allow_discrete_variables(::MySampler)`: Return `false` if your sampler needs every
  variable to be continuous, as the gradient-based ones do. `sample` checks the model against
  this before it starts.

- `Turing.Inference.gibbs_get_stats(::MyState)`: The statistics of the step that produced this
  state, as a `NamedTuple`, for a Gibbs chain to carry. Gibbs drops component transitions, so
  they cannot come from there. The wrapper implements this for you from
  `AbstractMCMC.getstats`.

- `Turing.Inference.post_sample_hook(chain, ::MySampler)`: Anything to report once sampling has
  finished, such as a warning about numerical errors. Returns `nothing` by default.

- `Turing.Inference.init_strategy(::MySampler)`: The `DynamicPPL.AbstractInitStrategy` your
  sampler starts from when the caller gives no `initial_params`. The default is
  `InitFromPrior()`; Gibbs asks each component for its own.

`supports_gibbs`, `allow_varying_dimension`, `allow_discrete_variables`, `init_strategy` and
`post_sample_hook` are the whole of Turing's own interface for running a sampler on a model. One that implements
`AbstractMCMC.step` for `DynamicPPL.Model` (option 1 above) and overrides whichever of them do
not match its defaults needs no wrapping at all; `externalsampler` exists only to supply the
`step` method option 2 leaves out. Serving as a Gibbs component takes two further methods,
`Turing.Inference.gibbs_get_parameter_values` and `Turing.Inference.gibbs_update_state!!`, which the
wrapper implements for you -- and which are also why `allow_varying_dimension` is not on the
list above. `allow_varying_dimension` is not forwarded, and the wrapper implements no
`gibbs_update_state!!` for a `ReshapedBlock`: the wrapped state is opaque, so there is no way to
tell what within it is shaped like the block. A wrapped sampler therefore cannot own a Gibbs
block whose set of variables changes, whatever it declares; one that needs to has to take
option 1.

One method the wrapper cannot implement for you:

- `AbstractMCMC.setparams!!(model::AbstractMCMC.LogDensityModel, state, params)`: required to
  serve as a Gibbs component. Gibbs re-conditions the model between sweeps, so this must
  recompute any log-density the state caches, not only write `params` into it. Define the
  three-argument form: AbstractMCMC's fallback drops `model` and calls a two-argument
  `setparams!!(state, params)`, which cannot recompute anything, and a state that caches a
  log-density would then start each step from the previous conditioning's value.
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

function allow_discrete_variables(spl::ExternalSampler)
    return allow_discrete_variables(spl.sampler)
end

# Where the sampler wants to start is its own business: wrapping does not constrain it, since
# `find_initial_params_ldf` turns whatever strategy it names into the vector the sampler sees.
# Contrast `allow_varying_dimension`, which the wrapper keeps at `false` whatever it wraps
# because there its own `gibbs_update_state!!` is the constraint.
init_strategy(spl::ExternalSampler) = init_strategy(spl.sampler)

# Same reasoning: whatever the sampler wants to report once sampling ends does not change for
# being wrapped.
function post_sample_hook(chain, spl::ExternalSampler; kwargs...)
    return post_sample_hook(chain, spl.sampler; kwargs...)
end

struct TuringState{S,P<:AbstractVector,L<:DynamicPPL.LogDensityFunction}
    state::S
    params::P
    ldf::L
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG, model::DynamicPPL.Model, sampler::ExternalSampler; kwargs...
)
    return _external_first_step(AbstractMCMC.step, rng, model, sampler; kwargs...)
end

function AbstractMCMC.step_warmup(
    rng::Random.AbstractRNG, model::DynamicPPL.Model, sampler::ExternalSampler; kwargs...
)
    return _external_first_step(AbstractMCMC.step_warmup, rng, model, sampler; kwargs...)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::ExternalSampler,
    state::TuringState;
    kwargs...,
)
    return _external_subsequent_step(
        AbstractMCMC.step, rng, model, sampler, state; kwargs...
    )
end

function AbstractMCMC.step_warmup(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::ExternalSampler,
    state::TuringState;
    kwargs...,
)
    return _external_subsequent_step(
        AbstractMCMC.step_warmup, rng, model, sampler, state; kwargs...
    )
end

# `step` and `step_warmup` differ only in which function is called on the wrapped
# sampler, so the four methods above delegate to these two helpers. The
# `step_function` argument should always be either `AbstractMCMC.step` or
# `AbstractMCMC.step_warmup`.
function _external_first_step(
    step_function::F,
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler_wrapper::ExternalSampler{unconstrained};
    initial_state=nothing,
    initial_params, # passed through from sample
    discard_sample=false,
    fix_transforms::Bool=false,
    kwargs...,
) where {F,unconstrained}
    sampler = sampler_wrapper.sampler

    # Construct LogDensityFunction
    tfm_strategy = unconstrained ? DynamicPPL.LinkAll() : DynamicPPL.UnlinkAll()
    f = DynamicPPL.LogDensityFunction(
        model,
        DynamicPPL.getlogjoint_internal,
        tfm_strategy;
        adtype=sampler_wrapper.adtype,
        fix_transforms=fix_transforms,
    )
    x = find_initial_params_ldf(rng, f, initial_params)

    # Then just call the inner step function with the right arguments.
    _, state_inner = if initial_state === nothing
        step_function(
            rng, AbstractMCMC.LogDensityModel(f), sampler; initial_params=x, kwargs...
        )

    else
        step_function(
            rng,
            AbstractMCMC.LogDensityModel(f),
            sampler,
            initial_state;
            initial_params=x,
            kwargs...,
        )
    end

    new_parameters = AbstractMCMC.getparams(f.model, state_inner)
    new_transition = if discard_sample
        nothing
    else
        new_stats = AbstractMCMC.getstats(state_inner)
        DynamicPPL.ParamsWithStats(new_parameters, f, new_stats)
    end

    return (new_transition, TuringState(state_inner, new_parameters, f))
end

function _external_subsequent_step(
    step_function::F,
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler_wrapper::ExternalSampler,
    state::TuringState;
    discard_sample=false,
    kwargs...,
) where {F}
    sampler = sampler_wrapper.sampler
    f = state.ldf

    # Then just call the inner step function with the right arguments.
    _, state_inner = step_function(
        rng, AbstractMCMC.LogDensityModel(f), sampler, state.state; kwargs...
    )

    new_parameters = AbstractMCMC.getparams(f.model, state_inner)
    new_transition = if discard_sample
        nothing
    else
        new_stats = AbstractMCMC.getstats(state_inner)
        DynamicPPL.ParamsWithStats(new_parameters, f, new_stats)
    end
    return (new_transition, TuringState(state_inner, new_parameters, f))
end

####
#### Gibbs interface
####

function gibbs_get_parameter_values(state::TuringState)
    pws = DynamicPPL.ParamsWithStats(
        state.params, state.ldf; include_log_probs=false, include_colon_eq=false
    )
    return pws.params
end

gibbs_get_stats(state::TuringState) = AbstractMCMC.getstats(state.state)

function gibbs_update_state!!(
    ::ExternalSampler,
    state::TuringState,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
)
    new_ldf, new_params, _ = gibbs_recompute_ldf_and_params(state.ldf, model, global_vals)
    # Update the inner sampler's state with the new parameters.
    new_inner_state = AbstractMCMC.setparams!!(
        AbstractMCMC.LogDensityModel(new_ldf), state.state, new_params
    )
    return TuringState(new_inner_state, new_params, new_ldf)
end
